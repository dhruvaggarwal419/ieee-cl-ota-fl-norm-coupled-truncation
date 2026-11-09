"""(D) The min-gain bottleneck.  beta = rho * min_k u_k, u_k = sqrt(P lam_k/(2e))/G_k,
so the device with the largest normalised gradient norm fixes the gain of everyone.

Scaling study over K and over gradient heterogeneity, plus an alternative gain
objective (quantile-q ceiling) that trades a small bias for a larger beta.
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otalib import gains, step_ota, E_ghat

P = 1.0
OUT = '/workspace/USER/r3ota/r3_D_bottleneck.json'


def make(K, hetero, seed=0, d=2):
    """hetero controls the spread of the device optima (hence of the gradient norms)."""
    r = np.random.default_rng(seed)
    nfar = max(1, K // 5)
    C = np.concatenate([r.normal(0, 0.3, (K - nfar, d)),
                        np.column_stack([r.normal(hetero, 0.3, nfar), r.normal(0, 0.3, (nfar, d - 1))])])
    lam = np.geomspace(0.25, 1.0, K); lam /= lam.mean(); r.shuffle(lam)
    return C, lam


def mse_terms(g, G, lam, rule, sigma, d, **kw):
    """analytic E||ghat - mean grad||^2 split into truncation variance, noise, bias^2."""
    K = len(G)
    a, c, beta, St = gains(rule, G, lam, P, **kw)
    p = np.exp(-(a * St) ** 2 / (P * lam))
    eff = a * c * p                       # expected received gain per device
    bias = ((eff[:, None] * g).sum(0) / (K * beta)) - g.mean(0)
    var_tr = float((((a * c) ** 2 * p - eff ** 2) * (G ** 2)).sum() / (K * beta) ** 2)
    var_n = float(d * sigma ** 2 / (K * beta) ** 2)
    return dict(beta=float(beta), bias2=float((bias ** 2).sum()), var_trunc=var_tr,
                var_noise=var_n, mse=float((bias ** 2).sum()) + var_tr + var_n,
                p_mean=float(p.mean()), umin=float((np.sqrt(P * lam / (2 * np.e)) / G).min()),
                Gmax=float(G.max()), Gmean=float(G.mean()))


def sim(rule, C, lam, seed=0, sigma=0.05, T=4000, tail=600, **kw):
    K, d = C.shape
    r = np.random.default_rng(seed); w = np.array([1.0] + [0.0] * (d - 1)); acc = []
    for t in range(T):
        g = w - C; G = np.linalg.norm(g, axis=1)
        gh, sent, a, beta, en = step_ota(r, g, G, lam, P, rule, sigma, d, **kw)
        w = w - min(0.5, 1.0 / (1 + t / 200)) * gh
        if not np.isfinite(w).all(): return np.full(d, np.nan)
        if t > T - tail: acc.append(w.copy())
    return np.mean(acc, 0)


if __name__ == '__main__':
    t0 = time.time(); R = {'P': P}
    SIG = 0.05; SEEDS = 5

    # ---- D1: scaling in K at fixed heterogeneity -------------------------
    R['D1_K'] = []
    for K in [5, 10, 20, 50, 100, 200]:
        C, lam = make(K, 3.0)
        wstar = C.mean(0)
        row = {'K': K}
        # analytic MSE at the start point (w0) and near the optimum
        for tagw, w in [('w0', np.array([1.0, 0.0])), ('wstar', wstar.copy())]:
            g = w - C; G = np.linalg.norm(g, axis=1)
            for rule, kw in [('prop', dict(rho=1.0)), ('fixed', dict(eta=0.5)),
                             ('blind', dict(rho=1.0, Gmax=float(np.linalg.norm(np.array([1.0, 0.]) - C, axis=1).max() * 1.1))),
                             ('quant50', dict(rho=1.0, q=0.5)), ('quant90', dict(rho=1.0, q=0.9))]:
                rr = 'quant' if rule.startswith('quant') else rule
                row[f'{tagw}_{rule}'] = mse_terms(g, G, lam, rr, SIG, 2, **kw)
        for rule, kw in [('prop', dict(rule='prop', rho=1.0)), ('fixed', dict(rule='fixed', eta=0.5)),
                         ('norm', dict(rule='norm', eta=0.5))]:
            e = [np.linalg.norm(sim(kw['rule'], C, lam, seed=s, sigma=SIG,
                                    **{k: v for k, v in kw.items() if k != 'rule'}) - wstar)
                 for s in range(SEEDS)]
            row[f'converr_{rule}'] = [float(np.mean(e)), float(np.std(e, ddof=1))]
        R['D1_K'].append(row)
        print('D1', K, time.time() - t0, flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)

    # ---- D2: scaling in gradient heterogeneity ---------------------------
    R['D2_hetero'] = []
    for het in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        K = 50; C, lam = make(K, het); wstar = C.mean(0)
        row = {'hetero': het, 'K': K}
        w = np.array([1.0, 0.0]); g = w - C; G = np.linalg.norm(g, axis=1)
        for rule, kw in [('prop', dict(rho=1.0)), ('fixed', dict(eta=0.5)),
                         ('quant50', dict(rho=1.0, q=0.5)), ('quant90', dict(rho=1.0, q=0.9))]:
            rr = 'quant' if rule.startswith('quant') else rule
            row[f'w0_{rule}'] = mse_terms(g, G, lam, rr, SIG, 2, **kw)
        for rule, kw in [('prop', dict(rule='prop', rho=1.0)), ('fixed', dict(rule='fixed', eta=0.5)),
                         ('norm', dict(rule='norm', eta=0.5))]:
            e = [np.linalg.norm(sim(kw['rule'], C, lam, seed=s, sigma=SIG,
                                    **{k: v for k, v in kw.items() if k != 'rule'}) - wstar)
                 for s in range(SEEDS)]
            row[f'converr_{rule}'] = [float(np.mean(e)), float(np.std(e, ddof=1))]
        R['D2_hetero'].append(row)
        print('D2', het, time.time() - t0, flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)

    # ---- D3: bias/variance trade-off curve over the quantile ceiling -----
    R['D3_tradeoff'] = []
    for K in [20, 100]:
        C, lam = make(K, 3.0); wstar = C.mean(0)
        for q in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            w = np.array([1.0, 0.0]); g = w - C; G = np.linalg.norm(g, axis=1)
            m = mse_terms(g, G, lam, 'quant', SIG, 2, rho=1.0, q=q)
            e = [np.linalg.norm(sim('quant', C, lam, seed=s, sigma=SIG, rho=1.0, q=q) - wstar)
                 for s in range(SEEDS)]
            m.update(K=K, q=q, converr=float(np.mean(e)), converr_sd=float(np.std(e, ddof=1)))
            R['D3_tradeoff'].append(m)
        print('D3', K, time.time() - t0, flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)

    # ---- D4: how beta tracks the shrinking gradients during training -----
    K = 20; C, lam = make(K, 3.0); wstar = C.mean(0)
    traj = {'round': [], 'Gmax': [], 'beta_prop': [], 'beta_blind': []}
    r = np.random.default_rng(0); w = np.array([1.0, 0.0])
    Gm0 = np.linalg.norm(w - C, axis=1).max() * 1.1
    for t in range(3000):
        g = w - C; G = np.linalg.norm(g, axis=1)
        gh, sent, a, beta, en = step_ota(r, g, G, lam, P, 'prop', SIG, 2, rho=1.0)
        if t % 50 == 0:
            _, _, bb, _ = gains('blind', G, lam, P, rho=1.0, Gmax=Gm0)
            traj['round'].append(t); traj['Gmax'].append(float(G.max()))
            traj['beta_prop'].append(float(beta)); traj['beta_blind'].append(float(bb))
        w = w - min(0.5, 1.0 / (1 + t / 200)) * gh
    R['D4_beta_traj'] = traj
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
