"""(A, part 3) logistic and small-net cases, redone properly.

In r3_A2 the common gain eta=0.5 left mean participation at 0.99, i.e. almost no
truncation, so nothing was being tested.  Here eta is CALIBRATED per scenario so
that the mean participation at the start point hits a target, and three targets
are swept.  Also reports distances to the ideal (noiseless, full participation)
converged point, which is the right reference for the non-convex case."""
import numpy as np, json, sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_A as A
from otalib import gains, step_ota

P = 1.0
OUT = '/workspace/USER/r3ota/r3_A_assumptions.json'


def calib_eta(G, lam, target):
    lo, hi = 1e-3, 50.0
    for _ in range(60):
        m = np.sqrt(lo * hi)
        p = np.exp(-m ** 2 * G ** 2 / (P * lam)).mean()
        if p > target: lo = m
        else: hi = m
    return float(np.sqrt(lo * hi))


def study(tag, model, X, Y, lam, target, T=3000, seeds=3, sigma=0.05, w0=None):
    npar = model.np
    w0 = np.zeros(npar) if w0 is None else np.asarray(w0, float)
    C = A.local_opt(model, X, Y, w0)

    def fg(wnp):
        g = A.dev_grads(model, torch.tensor(wnp), X, Y).numpy()
        return g, np.linalg.norm(g, axis=1)

    g0, G0 = fg(w0)
    eta = calib_eta(G0, lam, target)

    def sim(rule, seed, **kw):
        r = np.random.default_rng(seed); w = w0.copy(); acc = []; ps = []
        for t in range(T):
            g, G = fg(w)
            gh, sent, a, beta, en = step_ota(r, g, G, lam, P, rule, sigma, npar, **kw)
            ps.append(sent.mean())
            w = w - min(0.3, 1.0 / (1 + t / 200)) * gh
            if not np.isfinite(w).all(): return np.full(npar, np.nan), np.nan, np.nan
            if t > T - 400: acc.append(w.copy())
        wf = np.mean(acc, 0); g, G = fg(wf)
        return wf, float(np.linalg.norm(g.mean(0))), float(np.mean(ps))

    res = {}
    for name, rule, kw in [('ideal', 'ideal', {}), ('fixed', 'fixed', dict(eta=eta)),
                           ('prop', 'prop', dict(rho=1.0)), ('norm', 'norm', dict(eta=eta))]:
        o = [sim(rule, s, **kw) for s in range(seeds)]
        res[name] = dict(w=np.mean([x[0] for x in o], 0).tolist(),
                         gradnorm=float(np.mean([x[1] for x in o])),
                         gradnorm_sd=float(np.std([x[1] for x in o], ddof=1)) if seeds > 1 else 0.0,
                         part=float(np.mean([x[2] for x in o])))
    # mean-shift prediction with the bandwidth the theorem prescribes
    s2 = P * lam.mean() / (2 * eta ** 2)
    w = C.mean(0).copy()
    for _ in range(20000):
        k = np.exp(-((w - C) ** 2).sum(1) / (2 * s2))
        if k.sum() < 1e-300: break
        w = (k[:, None] * C).sum(0) / k.sum()
    ms = w
    wid = np.array(res['ideal']['w']); wf = np.array(res['fixed']['w'])
    eff = float(np.linalg.norm(C.mean(0) - ms))
    g, G = fg(wf)
    a, c, beta, St = gains('fixed', G, lam, P, eta=eta)
    p = np.exp(-(a * St) ** 2 / (P * lam))
    res.update(tag=tag, target_part=target, eta=eta, effect=eff,
               c_spread=float(np.linalg.norm(C - C.mean(0), axis=1).mean()),
               ms_pred_dist_from_cmean=eff,
               abs_T1=float(np.linalg.norm(wf - ms)),
               rel_T1=float(np.linalg.norm(wf - ms) / max(eff, 1e-12)),
               dist_fixed_ideal=float(np.linalg.norm(wf - wid)),
               dist_prop_ideal=float(np.linalg.norm(np.array(res['prop']['w']) - wid)),
               dist_norm_ideal=float(np.linalg.norm(np.array(res['norm']['w']) - wid)),
               bound4=float(((1 - p) * G).sum() / len(G)), mean_p_at_conv=float(p.mean()))
    print('%-28s target=%.2f eta=%.3f effect=%.4f relT1=%.3f  dist(fixed,ideal)=%.4f '
          'dist(prop,ideal)=%.4f dist(norm,ideal)=%.4f  gnorm ideal/prop/fixed/norm '
          '%.5f/%.5f/%.5f/%.5f part %.2f/%.2f' %
          (tag, target, eta, eff, res['rel_T1'], res['dist_fixed_ideal'],
           res['dist_prop_ideal'], res['dist_norm_ideal'], res['ideal']['gradnorm'],
           res['prop']['gradnorm'], res['fixed']['gradnorm'], res['norm']['gradnorm'],
           res['fixed']['part'], res['prop']['part']), flush=True)
    return res


if __name__ == '__main__':
    torch.set_num_threads(4)
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    K = 20
    X, Y, th = A.make_data(K=K, n=200, d=4, off=3.0, seed=0)
    lam1 = np.ones(K)
    lamh = np.geomspace(0.25, 4.0, K); lamh /= lamh.mean()
    nn_model = A.Net(4, 8)
    w0nn = np.random.default_rng(7).normal(0, 0.5, nn_model.np)

    R['A7_calibrated'] = []
    for tgt in [0.8, 0.5, 0.3]:
        R['A7_calibrated'].append(study('logistic, equal lambda', A.Logi(4), X, Y, lam1, tgt))
        json.dump(R, open(OUT, 'w'), indent=1)
        R['A7_calibrated'].append(study('logistic, 16x lambda spread', A.Logi(4), X, Y, lamh, tgt))
        json.dump(R, open(OUT, 'w'), indent=1)
        R['A7_calibrated'].append(study('tanh net h=8 (non-convex)', nn_model, X, Y, lam1,
                                        tgt, w0=w0nn))
        json.dump(R, open(OUT, 'w'), indent=1)
        print('target', tgt, 'done', time.time() - t0, flush=True)
    print('DONE', time.time() - t0, flush=True)
