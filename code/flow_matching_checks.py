"""
Numerical checks of the flow-matching derivation on a 2D, two-mode "action" distribution
(pass an obstacle on the left or on the right), using the exact optimal velocity field.

Convention: tau = 0 is noise, tau = 1 is data.
    x_tau = tau * A + (1 - tau) * eps,      eps ~ N(0, I)
    conditional target u = A - eps
    v*(x, tau) = (E[A | x_tau = x] - x) / (1 - tau)

Checks:
  1. L_CFM = L_FM + constant: the conditional loss ranks velocity fields exactly like the
     intractable marginal loss, and v* minimizes it.
  2. The analytic denoiser E[A | x_tau] (a mixture of Kalman updates) matches a
     brute-force Monte Carlo posterior mean.
  3. One Euler step lands every sample on E[A] (MSE regression); more steps recover both modes.
  4. Heun (2nd order) vs Euler at equal network evaluations.
  5. Instantaneous change of variables: log p_tau(x_tau) integrated along the ODE matches
     the closed-form marginal density.
"""

import numpy as np

rng = np.random.default_rng(0)

MU = np.array([[-1.6, 1.0], [1.6, 1.0]])   # two modes
W = np.array([0.5, 0.5])                     # mode weights
S = 0.2                                      # per-mode std (isotropic)
D = 2


# ---------------------------------------------------------------- target and exact field
def sample_data(n):
    k = rng.choice(2, size=n, p=W)
    return MU[k] + S * rng.standard_normal((n, D))


def log_gauss_iso(x, mean, var):
    d = x - mean
    return -0.5 * (d * d).sum(-1) / var - 0.5 * D * np.log(2 * np.pi * var)


def posterior_mean(x, tau):
    """E[A | x_tau = x] for the GMM prior: responsibilities x per-component Kalman update."""
    var = tau**2 * S**2 + (1 - tau) ** 2            # marginal variance of x_tau within a component
    gain = tau * S**2 / var                          # Kalman gain with H = tau I, R = (1-tau)^2 I
    logr = np.stack([np.log(W[k]) + log_gauss_iso(x, tau * MU[k], var) for k in range(2)], -1)
    logr -= logr.max(-1, keepdims=True)
    r = np.exp(logr); r /= r.sum(-1, keepdims=True)
    comp = np.stack([MU[k] + gain * (x - tau * MU[k]) for k in range(2)], 1)   # (n, 2, D)
    return (r[..., None] * comp).sum(1)


def v_star(x, tau):
    return (posterior_mean(x, tau) - x) / (1 - tau)


def log_p_tau(x, tau):
    var = tau**2 * S**2 + (1 - tau) ** 2
    ls = np.stack([np.log(W[k]) + log_gauss_iso(x, tau * MU[k], var) for k in range(2)], -1)
    m = ls.max(-1, keepdims=True)
    return (m + np.log(np.exp(ls - m).sum(-1, keepdims=True)))[:, 0]


# ---------------------------------------------------------------- 1. CFM vs FM
print("1) Conditional FM loss vs (intractable) marginal FM loss for several candidate fields")
n = 400_000
A = sample_data(n)
eps = rng.standard_normal((n, D))
tau = rng.uniform(0.0, 0.95, size=(n, 1))            # stay away from tau = 1 for stable MC
x = tau * A + (1 - tau) * eps
u = A - eps
# vectorized v* with a different tau per sample
var = tau**2 * S**2 + (1 - tau) ** 2
gain = tau * S**2 / var
logr = np.stack([np.log(W[k]) - 0.5 * ((x - tau * MU[k]) ** 2).sum(-1, keepdims=True) / var
                 - np.log(var) for k in range(2)], -1)[:, 0, :]
logr -= logr.max(-1, keepdims=True); r = np.exp(logr); r /= r.sum(-1, keepdims=True)
EA = sum(r[:, [k]] * (MU[k] + gain * (x - tau * MU[k])) for k in range(2))
vs = (EA - x) / (1 - tau)

candidates = {
    "v* (optimal)":            vs,
    "v* + 0.3 sin(x)":         vs + 0.3 * np.sin(x),
    "MSE field  E[A] - x":     (A.mean(0) - x) / (1 - tau),
    "zero field":              np.zeros_like(x),
}
print(f"   {'candidate':<22} {'L_CFM':>9} {'L_FM':>9} {'L_CFM - L_FM':>13}")
for name, v in candidates.items():
    lcfm = ((v - u) ** 2).sum(-1).mean()
    lfm = ((v - vs) ** 2).sum(-1).mean()
    print(f"   {name:<22} {lcfm:9.4f} {lfm:9.4f} {lcfm - lfm:13.4f}")

# ---------------------------------------------------------------- 2. denoiser vs Monte Carlo
print("\n2) Analytic E[A | x_tau] (mixture of Kalman updates) vs brute-force Monte Carlo")
prior = sample_data(2_000_000)
for x0, t0 in [((0.0, 0.0), 0.2), ((0.3, 0.4), 0.5), ((1.2, 0.9), 0.8)]:
    xq = np.array([x0])
    lw = log_gauss_iso(xq, t0 * prior, (1 - t0) ** 2)
    w = np.exp(lw - lw.max()); w /= w.sum()
    mc = (w[:, None] * prior).sum(0)
    an = posterior_mean(xq, t0)[0]
    print(f"   x={x0}, tau={t0}: analytic {np.round(an, 4)}   Monte Carlo {np.round(mc, 4)}")


# ---------------------------------------------------------------- 3 & 4. sampling
def sample_flow(n, steps, method="euler"):
    x = rng.standard_normal((n, D))
    h = 1.0 / steps
    for i in range(steps):
        t = i * h
        if method == "euler":
            x = x + h * v_star(x, t)
        else:                                          # Heun; last step falls back to Euler near tau = 1
            k1 = v_star(x, t)
            xe = x + h * k1
            t2 = min(t + h, 1 - 1e-4)
            x = x + 0.5 * h * (k1 + v_star(xe, t2))
    return x


def stats(x):
    left = (x[:, 0] < 0).mean()
    obstacle = ((np.abs(x[:, 0]) < 0.4) & (np.abs(x[:, 1] - 1.0) < 0.3)).mean()
    d = np.minimum(np.linalg.norm(x - MU[0], axis=1), np.linalg.norm(x - MU[1], axis=1))
    near = d < 3 * S
    spread = x[near & (x[:, 0] < 0)][:, 0].std() if (near & (x[:, 0] < 0)).sum() > 10 else np.nan
    return left, obstacle, d.mean(), spread


print("\n3) Euler sampling with the exact field (target: 50% left, 0% in obstacle, per-mode std 0.20)")
print(f"   {'steps':>5} {'left':>7} {'in obstacle':>12} {'dist to mode':>13} {'mode std':>9}")
for N in [1, 2, 4, 10, 40]:
    l, o, dm, sp = stats(sample_flow(20_000, N))
    print(f"   {N:>5} {l:7.2f} {o:12.2f} {dm:13.3f} {sp:9.3f}")

print("\n4) Heun vs Euler at equal network evaluations (NFE)")
print(f"   {'NFE':>4} {'method':>7} {'in obstacle':>12} {'dist to mode':>13}")
for nfe in [4, 10]:
    for method, steps in [("euler", nfe), ("heun", nfe // 2)]:
        l, o, dm, sp = stats(sample_flow(20_000, steps, method))
        print(f"   {nfe:>4} {method:>7} {o:12.3f} {dm:13.3f}")

# ---------------------------------------------------------------- 5. change of variables
print("\n5) log-density by integrating -div v along the ODE vs closed-form p_tau")
tau_end, steps, h_fd = 0.95, 4000, 1e-4
x = rng.standard_normal((5, D))
logp = log_gauss_iso(x, np.zeros(D), 1.0)
dt = tau_end / steps
for i in range(steps):
    t = i * dt
    div = np.zeros(len(x))
    for j in range(D):                                  # finite-difference trace of the Jacobian
        e = np.zeros(D); e[j] = h_fd
        div += (v_star(x + e, t)[:, j] - v_star(x - e, t)[:, j]) / (2 * h_fd)
    logp -= dt * div
    x = x + dt * v_star(x, t)
exact = log_p_tau(x, tau_end)
for a, b in zip(logp, exact):
    print(f"   ODE-integrated {a:9.4f}   closed form {b:9.4f}   diff {a - b:+.2e}")
