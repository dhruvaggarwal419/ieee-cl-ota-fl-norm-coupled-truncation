"""(A) Relax the Theorem-1 assumptions: unequal curvature, heterogeneous channels,
logistic (convex non-quadratic), small NN (non-convex).

Uniform metric for every scenario:
  c_k      = local optimum of device k (closed form for quadratics, numerical otherwise)
  ms_pred  = fixed point of the paper's Theorem-1 mean-shift map,
             w = sum_k kappa_k c_k / sum_k kappa_k,  kappa_k = exp(-||w-c_k||^2/(2 s^2)),
             s^2 = P lam_bar / (2 eta^2)      <-- the theorem AS STATED
  ex_pred  = fixed point of the exact expectation map E[ghat|w] = 0 (generalised,
             keeps per-device curvature / lambda)
  w_sim    = simulated converged point of the stochastic OTA run (fixed common gain)
  effect   = ||mean(c_k) - ms_pred||   (size of the bias the theorem predicts)
  rel_T1   = ||w_sim - ms_pred|| / effect   -> small means theorem still predicts
  rel_EX   = ||w_sim - ex_pred|| / effect
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otalib import gains, step_ota, E_ghat
import torch

P = 1.0
OUT = '/workspace/USER/r3ota/r3_A_assumptions.json'
ETA = 0.5
SEEDS = 5
T = 6000


# ---------------------------------------------------------------- quadratics
def make_quad(K=20, off=3.0, seed=0, cv=1.0, lamcv=1.0):
    r = np.random.default_rng(seed)
    nfar = max(1, K // 5)
    C = np.concatenate([r.normal(0, 0.3, (K - nfar, 2)),
                        np.column_stack([np.full(nfar, off), r.normal(0, 0.3, nfar)])])
    nu = np.geomspace(1.0 / cv, cv, K); nu = nu / nu.mean()
    r.shuffle(nu)
    lam = np.geomspace(1.0 / lamcv, lamcv, K); lam = lam / lam.mean()
    r.shuffle(lam)
    return C, nu, lam


def quad_grads(w, C, nu):
    g = nu[:, None] * (w - C)
    return g, np.linalg.norm(g, axis=1)


def sim_quad(rule, C, nu, lam, seed=0, sigma=0.05, T=T, tail=800, **kw):
    K, d = C.shape
    r = np.random.default_rng(seed); w = np.array([1.0, 0.0]); acc = []
    for t in range(T):
        g, G = quad_grads(w, C, nu)
        gh, sent, a, beta, en = step_ota(r, g, G, lam, P, rule, sigma, d, **kw)
        w = w - min(0.5, 1.0 / (1 + t / 200)) * gh
        if not np.isfinite(w).all():
            return np.full(d, np.nan)
        if t > T - tail: acc.append(w.copy())
    return np.mean(acc, 0)


def fp_exact_quad(rule, C, nu, lam, iters=20000, **kw):
    w = np.array([1.0, 0.0])
    for _ in range(iters):
        g, G = quad_grads(w, C, nu)
        w = w - 0.5 * E_ghat(g, G, lam, P, rule, **kw)
    return w


def fp_meanshift(C, s2, iters=20000, w0=None):
    w = np.array([1.0, 0.0]) if w0 is None else w0.copy()
    for _ in range(iters):
        k = np.exp(-((w - C) ** 2).sum(1) / (2 * s2))
        if k.sum() < 1e-300: return w
        w = (k[:, None] * C).sum(0) / k.sum()
    return w


def run_quad(tag, cv, lamcv, K=20, off=3.0):
    C, nu, lam = make_quad(K, off, 0, cv, lamcv)
    wtrue = (nu[:, None] * C).sum(0) / nu.sum()          # stationary point of F
    s2 = P * lam.mean() / (2 * ETA ** 2)                  # theorem as stated (nu=1, lam=lam_bar)
    ms = fp_meanshift(C, s2)
    ex = fp_exact_quad('fixed', C, nu, lam, eta=ETA)
    sims = np.array([sim_quad('fixed', C, nu, lam, seed=s, eta=ETA) for s in range(SEEDS)])
    w_sim = sims.mean(0)
    props = np.array([sim_quad('prop', C, nu, lam, seed=s, rho=1.0) for s in range(SEEDS)])
    norms = np.array([sim_quad('norm', C, nu, lam, seed=s, eta=ETA) for s in range(SEEDS)])
    effect = float(np.linalg.norm(C.mean(0) - ms))
    return dict(tag=tag, cv=cv, lamcv=lamcv, K=K,
                w_true=wtrue.tolist(), c_mean=C.mean(0).tolist(),
                ms_pred=ms.tolist(), ex_pred=ex.tolist(),
                w_sim_fixed=w_sim.tolist(), w_sim_fixed_std=sims.std(0, ddof=1).tolist(),
                w_sim_prop=props.mean(0).tolist(), w_sim_norm=norms.mean(0).tolist(),
                effect=effect,
                abs_T1=float(np.linalg.norm(w_sim - ms)),
                abs_EX=float(np.linalg.norm(w_sim - ex)),
                rel_T1=float(np.linalg.norm(w_sim - ms) / max(effect, 1e-12)),
                rel_EX=float(np.linalg.norm(w_sim - ex) / max(effect, 1e-12)),
                err_prop=float(np.linalg.norm(props.mean(0) - wtrue)),
                err_norm=float(np.linalg.norm(norms.mean(0) - wtrue)),
                err_fixed=float(np.linalg.norm(w_sim - wtrue)))


# ------------------------------------------------- torch models (A3, A4)
def make_data(K=20, n=200, d=4, off=3.0, seed=0):
    r = np.random.default_rng(seed)
    nfar = max(1, K // 5)
    th = np.concatenate([r.normal(0, 0.3, (K - nfar, d)),
                         np.column_stack([np.full(nfar, off), r.normal(0, 0.3, (nfar, d - 1))])])
    X = r.normal(0, 1, (K, n, d))
    logit = np.einsum('knd,kd->kn', X, th)
    Y = (r.random((K, n)) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return torch.tensor(X), torch.tensor(Y), th


class Logi:
    def __init__(s, d, gamma=0.05): s.d = d; s.gamma = gamma; s.np = d
    def loss(s, w, X, Y):
        z = X @ w
        return torch.nn.functional.binary_cross_entropy_with_logits(
            z, Y, reduction='none').mean(-1) + 0.5 * s.gamma * (w ** 2).sum()


class Net:
    def __init__(s, d, h=8, gamma=0.05):
        s.d = d; s.h = h; s.gamma = gamma; s.np = d * h + h + h + 1
    def unpack(s, w):
        i = 0; W1 = w[i:i + s.d * s.h].reshape(s.d, s.h); i += s.d * s.h
        b1 = w[i:i + s.h]; i += s.h; W2 = w[i:i + s.h]; i += s.h; b2 = w[i]
        return W1, b1, W2, b2
    def loss(s, w, X, Y):
        W1, b1, W2, b2 = s.unpack(w)
        z = torch.tanh(X @ W1 + b1) @ W2 + b2
        return torch.nn.functional.binary_cross_entropy_with_logits(
            z, Y, reduction='none').mean(-1) + 0.5 * s.gamma * (w ** 2).sum()


def dev_grads(model, w, X, Y):
    """per-device gradients at a common w, vectorised with vmap."""
    from torch.func import grad, vmap
    f = lambda wk, x, y: model.loss(wk, x[None], y[None]).sum()
    return vmap(grad(f), in_dims=(None, 0, 0))(w, X, Y)


def local_opt(model, X, Y, w0, steps=3000):
    """K local optima in parallel: one (K, npar) tensor, per-device loss."""
    from torch.func import grad, vmap
    K = X.shape[0]
    W = torch.tensor(np.tile(w0, (K, 1)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([W], lr=0.05)
    f = lambda wk, x, y: model.loss(wk, x[None], y[None]).sum()
    for t in range(steps):
        opt.zero_grad()
        W.grad = vmap(grad(f), in_dims=(0, 0, 0))(W, X, Y)
        opt.step()
    return W.detach().numpy().copy()


def run_model(tag, model, X, Y, lam, sigma=0.05, T=3000, seeds=3, eta=ETA, w0=None):
    npar = model.np
    w0 = np.zeros(npar) if w0 is None else np.asarray(w0, float)
    C = local_opt(model, X, Y, w0)

    def full_grad(wnp):
        g = dev_grads(model, torch.tensor(wnp), X, Y).numpy()
        return g, np.linalg.norm(g, axis=1)

    def sim(rule, seed, **kw):
        r = np.random.default_rng(seed); w = w0.copy(); acc = []
        for t in range(T):
            g, G = full_grad(w)
            gh, sent, a, beta, en = step_ota(r, g, G, lam, P, rule, sigma, npar, **kw)
            w = w - min(0.3, 1.0 / (1 + t / 200)) * gh
            if not np.isfinite(w).all(): return np.full(npar, np.nan), np.nan
            if t > T - 400: acc.append(w.copy())
        wf = np.mean(acc, 0)
        g, G = full_grad(wf)
        return wf, float(np.linalg.norm(g.mean(0)))

    res = {}
    for name, kw in [('ideal', dict(rule='ideal')), ('fixed', dict(rule='fixed', eta=eta)),
                     ('prop', dict(rule='prop', rho=1.0)), ('norm', dict(rule='norm', eta=eta))]:
        ws, gn = zip(*[sim(kw['rule'], s, **{k: v for k, v in kw.items() if k != 'rule'})
                       for s in range(seeds)])
        res[name] = dict(w=np.mean(ws, 0).tolist(), gradnorm=float(np.mean(gn)),
                         gradnorm_std=float(np.std(gn, ddof=1)) if seeds > 1 else 0.0)
    # mean-shift prediction in parameter space
    s2 = P * lam.mean() / (2 * eta ** 2)
    w = C.mean(0).copy()
    for _ in range(5000):
        k = np.exp(-((w - C) ** 2).sum(1) / (2 * s2))
        if k.sum() < 1e-300: break
        w = (k[:, None] * C).sum(0) / k.sum()
    ms = w
    # true stationary point of F (from the same init)
    wstar = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([wstar], lr=0.05)
    for t in range(6000):
        opt.zero_grad(); l = model.loss(wstar, X, Y).sum(); l.backward(); opt.step()
    wstar = wstar.detach().numpy()
    wsf = np.array(res['fixed']['w'])
    effect = float(np.linalg.norm(C.mean(0) - ms))
    # bound (4) check at the fixed-gain fixed point
    g, G = full_grad(wsf)
    a, c, beta, St = gains('fixed', G, lam, P, eta=eta)
    p = np.exp(-(a * St) ** 2 / (P * lam))
    bound = float(((1 - p) * G).sum() / len(G))
    res.update(tag=tag, c_mean=C.mean(0).tolist(), ms_pred=ms.tolist(), wstar=wstar.tolist(),
               effect=effect, abs_T1=float(np.linalg.norm(wsf - ms)),
               rel_T1=float(np.linalg.norm(wsf - ms) / max(effect, 1e-12)),
               dist_fixed_wstar=float(np.linalg.norm(wsf - wstar)),
               dist_prop_wstar=float(np.linalg.norm(np.array(res['prop']['w']) - wstar)),
               dist_norm_wstar=float(np.linalg.norm(np.array(res['norm']['w']) - wstar)),
               dist_ideal_wstar=float(np.linalg.norm(np.array(res['ideal']['w']) - wstar)),
               gradnorm_bound=bound, mean_p=float(p.mean()))
    return res


if __name__ == '__main__':
    t0 = time.time(); R = {'eta': ETA, 'P': P, 'seeds': SEEDS, 'T': T}
    R['A0_baseline'] = run_quad('equal curvature, equal lambda', 1.0, 1.0)
    print('A0', time.time() - t0, flush=True)
    R['A1_curvature'] = [run_quad(f'curv cv={cv}', cv, 1.0) for cv in [1.0, 1.5, 2.0, 4.0, 8.0]]
    print('A1', time.time() - t0, flush=True)
    R['A2_channel'] = [run_quad(f'lam cv={c}', 1.0, c) for c in [1.0, 2.0, 4.0, 10.0, 40.0]]
    print('A2', time.time() - t0, flush=True)
    R['A3_both'] = [run_quad(f'both cv={c}', c, c) for c in [2.0, 4.0]]
    json.dump(R, open(OUT, 'w'), indent=1)
    print('A3comb', time.time() - t0, flush=True)

    print('DONE (quadratic part; run r3_A2.py for the model part)', time.time() - t0, flush=True)
