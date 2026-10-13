"""
Split-K (FlashDecoding-style) decode attention: a NumPy reference.

Mirrors the two-stage GPU kernel:
  stage 1: each "CTA" processes one chunk of the KV sequence with an online softmax,
           producing a partial state (m, l, acc)
  stage 2: partial states are merged with an associative log-sum-exp combine

It checks that:
  1. split-K output == standard softmax attention, for any number of splits
  2. the merge is order-independent (associative), so tree / multi-GPU reductions are exact
  3. a simple heuristic for choosing the number of splits S fills the SMs

GQA packing is included: g query heads share one KV head, so Q_g is (g, d).
"""

import numpy as np
from functools import reduce


# ---------- reference ----------

def reference_attention(q, K, V):
    """q: (g, d), K, V: (T, d) -> (g, d). Standard softmax attention for one KV head."""
    d = q.shape[-1]
    s = q @ K.T / np.sqrt(d)                     # (g, T)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ V


# ---------- stage 1: one CTA over one KV chunk, tiled with an online softmax ----------

def chunk_partial(q, Kc, Vc, tile=128):
    """Online softmax over a KV chunk, processed tile by tile as a GPU CTA would.
    Returns the partial state (m, l, acc): running max, running sum, unnormalized output."""
    g, d = q.shape
    m = np.full(g, -np.inf)
    l = np.zeros(g)
    acc = np.zeros((g, d))
    for t0 in range(0, Kc.shape[0], tile):
        Kt, Vt = Kc[t0:t0 + tile], Vc[t0:t0 + tile]   # "load tile HBM -> SRAM"
        s = q @ Kt.T / np.sqrt(d)                      # (g, T_tile): all g queries reuse the tile
        m_new = np.maximum(m, s.max(axis=1))
        scale = np.exp(m - m_new)                      # rescale old state to the new max
        e = np.exp(s - m_new[:, None])
        l = l * scale + e.sum(axis=1)
        acc = acc * scale[:, None] + e @ Vt
        m = m_new
    return m, l, acc


# ---------- stage 2: associative merge ----------

def combine(a, b):
    """(m1,l1,a1) ⊕ (m2,l2,a2): exact, associative, order-independent."""
    m1, l1, a1 = a
    m2, l2, a2 = b
    m = np.maximum(m1, m2)
    w1, w2 = np.exp(m1 - m), np.exp(m2 - m)
    return m, l1 * w1 + l2 * w2, a1 * w1[:, None] + a2 * w2[:, None]


def split_k_decode(q, K, V, n_splits, order=None):
    """Full split-K decode: partition the sequence, compute partials, merge."""
    T = K.shape[0]
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    partials = [chunk_partial(q, K[s:e], V[s:e]) for s, e in zip(bounds[:-1], bounds[1:])]
    if order is not None:                             # merge in an arbitrary order
        partials = [partials[i] for i in order]
    m, l, acc = reduce(combine, partials)
    return acc / l[:, None]


def lse_merge(partials):
    """Equivalent stage-2 form used by many kernels: store normalized O_s and lse_s per split."""
    O = [acc / l[:, None] for m, l, acc in partials]
    lse = np.stack([m + np.log(l) for m, l, acc in partials])   # (S, g)
    lse_star = np.log(np.exp(lse - lse.max(0)).sum(0)) + lse.max(0)
    w = np.exp(lse - lse_star)                                    # (S, g)
    return sum(w[s][:, None] * O[s] for s in range(len(O)))


# ---------- choosing S ----------

def choose_splits(batch, kv_heads, seq_len, n_sm=132, waves=1, min_chunk=512):
    """Fill about `waves` full waves of SMs, without chunks smaller than min_chunk tokens."""
    base = batch * kv_heads
    s = max(1, int(np.ceil(waves * n_sm / base)))
    s = min(s, max(1, seq_len // min_chunk))
    return s


# ---------- checks ----------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g, d, T = 8, 128, 32_768          # GQA group of 8 query heads, head dim 128, 32k context
    q = rng.standard_normal((g, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))

    ref = reference_attention(q, K, V)

    print("1) split-K vs reference attention")
    for S in [1, 4, 16, 64]:
        out = split_k_decode(q, K, V, S)
        print(f"   S={S:>3}: max |error| = {np.abs(out - ref).max():.2e}")

    print("\n2) merge order independence (associativity)")
    S = 16
    order = rng.permutation(S)
    out_shuffled = split_k_decode(q, K, V, S, order=order)
    print(f"   shuffled merge order: max |error| = {np.abs(out_shuffled - ref).max():.2e}")

    bounds = np.linspace(0, T, S + 1).astype(int)
    parts = [chunk_partial(q, K[s:e], V[s:e]) for s, e in zip(bounds[:-1], bounds[1:])]
    print(f"   LSE-form merge:       max |error| = {np.abs(lse_merge(parts) - ref).max():.2e}")

    print("\n3) choosing S on a 132-SM GPU (KV heads = 8)")
    print(f"   {'batch':>5} {'context':>8} {'S':>4} {'CTAs':>6} {'chunk':>8}")
    for B, L in [(1, 4_096), (1, 131_072), (4, 131_072), (16, 32_768), (64, 8_192)]:
        S = choose_splits(B, 8, L)
        print(f"   {B:>5} {L:>8} {S:>4} {B * 8 * S:>6} {L // S:>8}")

    print("\n   Partial-state overhead vs KV bytes per head (FP16, 128k ctx, S=17):")
    S, L = 17, 131_072
    partial_bytes = S * g * (d + 2) * 4          # acc + m + l in fp32
    kv_bytes = 2 * L * d * 2
    print(f"   partials {partial_bytes / 1e3:.0f} KB  vs  KV {kv_bytes / 1e6:.0f} MB "
          f"({100 * partial_bytes / kv_bytes:.3f}%)")
