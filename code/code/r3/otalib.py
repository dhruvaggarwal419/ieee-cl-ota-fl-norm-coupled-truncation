"""Core OTA truncated-channel-inversion mechanism, shared by all r3 experiments.

Model (paper eq. 1-2):
  device k transmits  (a_k/h_k) * s_k   with payload s_k = c_k * g_k
  energy  a_k^2 ||s_k||^2 / |h_k|^2 <= P   -> silence otherwise
  server  y = sum_k 1_k a_k c_k g_k + n ,  ghat = y / (K beta)
  unbiased  <=>  a_k c_k p_k = beta  for all k,  p_k = exp(-a_k^2 Stest_k^2/(P lam_k))

Stest_k is the norm the DEVICE uses in its silence test.  For honest rules
Stest_k = ||s_k||.  For the Abrar&Michelusi norm-blind rule Stest_k = Gmax >= ||s_k||
(the device pretends its norm is the bound, which is conservative and keeps p_k
known to the server without per-round signalling).
"""
import numpy as np
from scipy.special import lambertw

SQ2E = np.sqrt(2 * np.e)


def lw_x(r):
    """smaller root x of sqrt(x) e^{-x} = r  (r <= 1/sqrt(2e)), x <= 1/2."""
    r = np.minimum(np.asarray(r, dtype=float), 1.0 / SQ2E - 1e-12)
    return -lambertw(-2 * r ** 2, 0).real / 2


def a_from_beta(beta, S, lam, P):
    """gain a with a*exp(-a^2 S^2/(P lam)) = beta  (smaller root)."""
    r = beta * S / np.sqrt(P * lam)
    return np.sqrt(lw_x(r) * P * lam) / S


def a_cheap(beta, S, lam, P, iters=1):
    """Lambert-W-free approximation: fixed point a = beta*exp(a^2 S^2/(P lam))."""
    a = np.asarray(np.broadcast_to(beta, np.shape(S)), dtype=float).copy()
    for _ in range(iters):
        a = beta * np.exp(np.minimum(a ** 2 * S ** 2 / (P * lam), 0.5))
    return a


def gains(rule, G, lam, P, eta=None, rho=1.0, Gmax=None, q=0.0, clipC=None,
          Gdesign=None, cheap=False, betafix=None, cvec=None):
    """Return (a, c, beta, Stest).  G = true payload-relevant gradient norms.
    Gdesign = norms the rule is allowed to use (imperfect info experiments)."""
    K = len(G)
    Gd = G if Gdesign is None else np.asarray(Gdesign, float)
    ones = np.ones(K)
    if rule == 'ideal':
        return ones, ones, 1.0, np.zeros(K)
    if rule in ('fixed', 'sched'):
        return np.full(K, eta), ones, float(eta), G
    if rule == 'cfixed':                    # generic payload weighting (q-FedAvg over the air)
        c = np.asarray(cvec, float) * K / np.sum(cvec)
        return np.full(K, eta), c, float(eta), c * G
    if rule == 'clip':
        c = np.minimum(1.0, clipC / np.maximum(G, 1e-12))
        return np.full(K, eta), c, float(eta), c * G
    if rule == 'norm':                      # normalise payload: p_k norm-free
        c = 1.0 / np.maximum(G, 1e-12)
        return np.full(K, eta), c, float(eta), ones
    if rule == 'norm_ub':                   # normalise + unbiased prescaler over lam
        c = 1.0 / np.maximum(G, 1e-12)
        u = np.sqrt(P * lam / (2 * np.e))
        beta = rho * u.min()
        a = a_cheap(beta, ones, lam, P) if cheap else a_from_beta(beta, ones, lam, P)
        return a, c, float(beta), ones
    if rule == 'blind':                     # Abrar & Michelusi fixed Gmax
        Sb = np.full(K, float(Gmax))
        u = np.sqrt(P * lam / (2 * np.e)) / Sb
        beta = rho * u.min()
        a = a_cheap(beta, Sb, lam, P) if cheap else a_from_beta(beta, Sb, lam, P)
        return a, ones, float(beta), Sb
    if rule in ('prop', 'quant'):
        # beta is set by the SERVER from the reported (possibly imperfect) norms Gd;
        # a_k is set by the DEVICE, which always knows its own true G_k and lam_k.
        u_rep = np.sqrt(P * lam / (2 * np.e)) / np.maximum(Gd, 1e-12)
        u_true = np.sqrt(P * lam / (2 * np.e)) / np.maximum(G, 1e-12)
        beta = rho * (u_rep.min() if q <= 0 else float(np.quantile(u_rep, q)))
        if betafix is not None:
            beta = float(betafix)
        b = np.minimum(beta, u_true)        # devices below the ceiling use their max gain
        a = a_cheap(b, G, lam, P) if cheap else a_from_beta(b, G, lam, P)
        return a, ones, float(beta), G
    raise ValueError(rule)


def step_ota(rng, g, G, lam, P, rule, sigma, d, fix_test_norm=None, **kw):
    """One OTA round.  Returns (ghat, p_realised_mask, a, beta, energy_used)."""
    K = len(G)
    a, c, beta, Stest = gains(rule, G, lam, P, **kw)
    if rule == 'ideal':
        return g.mean(0), np.ones(K, bool), a, 1.0, np.zeros(K)
    St = Stest if fix_test_norm is None else np.full(K, fix_test_norm)
    h2 = rng.exponential(lam)
    need = (a * St) ** 2 / P
    sent = h2 >= need
    y = ((sent * a * c)[:, None] * g).sum(0) + rng.normal(0, sigma, d)
    if rule == 'sched':                     # perfect activity detection at the server
        ghat = y / (max(sent.sum(), 1) * beta)
    else:
        ghat = y / (K * beta)
    energy = np.where(sent, (a * c * G) ** 2 / np.maximum(h2, 1e-30), 0.0)
    return ghat, sent, a, beta, energy


def E_ghat(g, G, lam, P, rule, **kw):
    """Exact E[ghat | w] (no sampling)."""
    K = len(G)
    a, c, beta, Stest = gains(rule, G, lam, P, **kw)
    if rule == 'ideal':
        return g.mean(0)
    p = np.exp(-(a * Stest) ** 2 / (P * lam))
    return ((a * c * p)[:, None] * g).sum(0) / (K * beta)
