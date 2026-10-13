"""
Numerical checks of MoE gating claims: softmax vs sigmoid, with and without top-k renormalization.

Checks:
  1. Top-k selection is identical for softmax and sigmoid (both are monotonic in the logit).
  2. Output scale: how much sum(gates) varies from token to token under each scheme.
  3. Shift invariance: effect of adding a constant to every logit.
  4. Softmax top-k mass shrinks as the number of experts E grows; renormalization removes that.
  5. Router gradients (finite differences): which logits get gradient, and whether the
     common "scale" direction does. At k = 1, renormalized gates give zero router gradient.
  6. Balancing bias: changes which experts are selected, but the renormalized sigmoid
     gates are computed from unbiased scores, so mixture weights are not distorted.
"""

import numpy as np

rng = np.random.default_rng(0)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def gates(s, k, scheme, bias=None):
    """Return (selected indices, gate values) for one token's logits s (shape E)."""
    sel_score = s if bias is None else s + bias           # bias affects selection only
    idx = np.argsort(-sel_score)[:k]
    if scheme == "softmax_raw":
        g = softmax(s)[idx]
    elif scheme == "softmax_renorm":
        g = softmax(s[idx])                                # == renormalized top-k softmax
    elif scheme == "sigmoid_raw":
        g = sigmoid(s[idx])
    elif scheme == "sigmoid_renorm":
        g = sigmoid(s[idx]); g = g / g.sum()
    return idx, g


SCHEMES = ["softmax_raw", "softmax_renorm", "sigmoid_raw", "sigmoid_renorm"]

# ---------------------------------------------------------------- 1. selection identity
E, k, T = 256, 8, 2000
logits = rng.normal(0, 1.0, size=(T, E)) * rng.uniform(0.5, 2.0, size=(T, 1))   # tokens differ in sharpness
same = all(set(np.argsort(-softmax(s))[:k]) == set(np.argsort(-sigmoid(s))[:k]) for s in logits)
print(f"1) Top-{k} selection identical for softmax and sigmoid on {T} tokens: {same}")

# ---------------------------------------------------------------- 2. output scale variability
print(f"\n2) Σ gates per token (E={E}, k={k}); this is the MoE output's scale")
print(f"   {'scheme':<16} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}")
for sc in SCHEMES:
    sums = np.array([gates(s, k, sc)[1].sum() for s in logits])
    print(f"   {sc:<16} {sums.mean():7.3f} {sums.std():7.3f} {sums.min():7.3f} {sums.max():7.3f}")

# ---------------------------------------------------------------- 3. shift invariance
s0 = logits[0]
print("\n3) Σ gates for one token as every logit is shifted by c")
print(f"   {'scheme':<16}" + "".join(f"{'c=' + str(c):>9}" for c in [-3, 0, 3]))
for sc in SCHEMES:
    vals = [gates(s0 + c, k, sc)[1].sum() for c in [-3, 0, 3]]
    print(f"   {sc:<16}" + "".join(f"{v:9.3f}" for v in vals))

# ---------------------------------------------------------------- 4. E scaling
print(f"\n4) Mean top-{k} softmax mass (raw) vs number of experts E")
for E_ in [8, 32, 128, 256, 1024]:
    if E_ < k:
        continue
    L = rng.normal(0, 1.0, size=(500, E_))
    mass = np.mean([np.sort(softmax(s))[-k:].sum() for s in L])
    print(f"   E={E_:>5}: {mass:.3f}   (renormalized: 1.000)")

# ---------------------------------------------------------------- 5. router gradients
print("\n5) Router gradient of a scalar loss through the gates (finite differences)")
E5, d = 16, 4
W_exp = rng.normal(size=(E5, d))            # stand-in expert outputs E_i(x) for one token
target = rng.normal(size=d)
s5 = rng.normal(size=E5)


def loss(s, k, sc):
    idx, g = gates(s, k, sc)
    y = (g[:, None] * W_exp[idx]).sum(0)
    return 0.5 * np.sum((y - target) ** 2)


def grad(s, k, sc, h=1e-6):
    gr = np.zeros_like(s)
    for i in range(len(s)):
        e = np.zeros_like(s); e[i] = h
        gr[i] = (loss(s + e, k, sc) - loss(s - e, k, sc)) / (2 * h)
    return gr


print(f"   {'scheme':<16} {'k':>2} {'|grad| selected':>16} {'|grad| unselected':>18} {'common-mode':>12}")
for kk in [1, 4]:
    idx_sel = np.argsort(-s5)[:kk]
    mask = np.zeros(E5, bool); mask[idx_sel] = True
    for sc in SCHEMES:
        g_ = grad(s5, kk, sc)
        # common-mode: directional derivative along adding the same constant to all selected logits
        cm = g_[mask].sum()
        print(f"   {sc:<16} {kk:>2} {np.abs(g_[mask]).sum():16.4f} {np.abs(g_[~mask]).sum():18.4f} {cm:12.4f}")

# ---------------------------------------------------------------- 6. balancing bias
print("\n6) Bias for load balancing (DeepSeek-V3 style): selection on σ(s)+b, gates from σ(s) renormalized")
s6 = rng.normal(size=E5)
b = np.zeros(E5); hot = np.argsort(-s6)[0]; b[hot] = -5.0          # push the hottest expert out
idx_nb, g_nb = gates(s6, 4, "sigmoid_renorm")
idx_b, g_b = gates(s6, 4, "sigmoid_renorm", bias=b)
print(f"   selected without bias: {sorted(idx_nb.tolist())}")
print(f"   selected with bias   : {sorted(idx_b.tolist())}   (expert {hot} pushed out)")
common = sorted(set(idx_nb) & set(idx_b))
ratio_nb = {i: g for i, g in zip(idx_nb, g_nb)}
ratio_b = {i: g for i, g in zip(idx_b, g_b)}
a, c = common[0], common[1]
print(f"   relative weight of experts {a}:{c} without bias {ratio_nb[a] / ratio_nb[c]:.4f}, "
      f"with bias {ratio_b[a] / ratio_b[c]:.4f}  (unchanged: bias never enters the gate)")
