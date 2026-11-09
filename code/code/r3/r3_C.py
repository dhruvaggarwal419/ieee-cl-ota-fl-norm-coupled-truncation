"""(C) Empirical convergence behaviour: loss and gradient norm vs round, with
confidence intervals, on a convex objective (MNIST softmax regression) and a
non-convex one (Fashion-MNIST CNN).  Also the measured variance of the aggregate.
No theorem is claimed here - this is evidence only.

Every method gets its own tuned step size (and gain parameter where it has one),
tuned on mean device TRAINING loss, so no test data is used for tuning."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, pick_sigma, tune

OUT = '/workspace/USER/r3ota/r3_C_convergence.json'
P = 1.0


def ci95(x):
    from scipy.stats import t as tdist
    x = np.asarray(x, float); n = len(x)
    if n < 2: return float(x.mean()), 0.0
    return float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def rate_fit(rounds, vals):
    r = np.asarray(rounds, float); v = np.asarray(vals, float)
    m = (r > max(r.max() / 4, 1)) & (v > 0)
    if m.sum() < 4: return None
    return float(np.polyfit(np.log(r[m]), np.log(v[m]), 1)[0])


def study(tag, task, lam, T, sigma, seeds, methods, log_every):
    out = {}
    for name, m, lr, kw in methods:
        runs = [fl_run(task, m, lr, T, sigma, lam, seed=s, log_every=log_every, **kw)
                for s in range(seeds)]
        if any(r['diverged'] for r in runs):
            out[name] = dict(diverged=int(sum(r['diverged'] for r in runs)), lr=lr)
            print(tag, name, 'DIVERGED', flush=True); continue
        L = np.array([r['log']['floss'] for r in runs])
        Gn = np.array([r['log']['gnorm'] for r in runs])
        MS = np.array([r['log']['mse'] for r in runs])
        rd = runs[0]['log']['round']
        best = np.minimum.accumulate(Gn ** 2, axis=1)
        out[name] = dict(
            lr=lr, par=kw,
            rounds=rd,
            loss_mean=L.mean(0).tolist(), loss_ci=[ci95(L[:, j])[1] for j in range(L.shape[1])],
            gnorm_mean=Gn.mean(0).tolist(), gnorm_ci=[ci95(Gn[:, j])[1] for j in range(Gn.shape[1])],
            mse_mean=MS.mean(0).tolist(),
            final_loss=ci95(L[:, -1]), final_gnorm=ci95(Gn[:, -1]),
            final_acc=ci95([r['acc'] for r in runs]),
            worst_rec=ci95([r['worst_rec'] for r in runs]),
            rate_loss=rate_fit(rd, L.mean(0) - L.mean(0).min() + 1e-6),
            rate_bestgrad2=rate_fit(rd, best.mean(0)),
            agg_mse_mean=float(MS.mean()), agg_mse_last=float(MS[:, -1].mean()),
            part=float(np.mean([r['part'] for r in runs])),
            beta_mean=float(np.mean([np.mean(r['log']['beta']) for r in runs])),
            diverged=0)
        print(tag, name, 'lr=%g loss=%.4f acc=%.4f worst=%.3f aggmse=%.4g' %
              (lr, out[name]['final_loss'][0], out[name]['final_acc'][0],
               out[name]['worst_rec'][0], out[name]['agg_mse_mean']), flush=True)
    return out


def build(task, lam, sigma, Gmax, lrs_mean, lrs_unit, etas, Ttune):
    """tune each method, return the method list for study()."""
    spec = [('ideal', 'ideal', lrs_mean, (None,), None, {}),
            ('common_gain', 'fixed', lrs_mean, etas, 'eta', {}),
            ('proposed', 'prop', lrs_mean, (None,), None, dict(rho=1.0)),
            ('norm_blind_Gmax', 'blind', lrs_mean, (None,), None, dict(rho=1.0, Gmax=Gmax)),
            ('normalised', 'norm', lrs_unit, etas, 'eta', {}),
            ('normalised_unbiased', 'norm_ub', lrs_unit, (None,), None, dict(rho=1.0))]
    meth, tuned = [], {}
    for name, m, lrs, pars, pn, kw in spec:
        v, lr, p = tune(task, m, lam, Ttune, sigma, lrs, pars, pn, **kw)
        k = dict(kw)
        if pn is not None and p is not None: k[pn] = p
        meth.append((name, m, lr, k)); tuned[name] = dict(loss=v, lr=lr, par=p)
        print('tuned', name, lr, p, '%.4f' % v, flush=True)
    return meth, tuned


if __name__ == '__main__':
    t0 = time.time(); R = {}
    SEEDS = 5

    # ---------------- convex: MNIST softmax, minority split ----------------
    task = Task('mnist_softmax', K=20, seed=0, nper=595, ncls=7, minority=([7, 8, 9], 3))
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    pil = fl_run(task, 'prop', 2.0, 60, 0.0, lam, seed=99, rho=1.0, log_every=1)
    Gmax = float(np.array(pil['log']['G']).max() * 1.1)
    R['convex_Gmax'] = Gmax; R['convex_d'] = task.d
    for tgt, nm in [(0.25, 'lownoise'), (4.0, 'highnoise')]:
        sg, gb, b0 = pick_sigma(task, lam, 2.0, target=tgt)
        R[f'convex_sigma_{nm}'] = dict(sigma=sg, gnorm0=gb, beta0=b0, noise_to_grad=tgt)
        meth, tuned = build(task, lam, sg, Gmax, [0.5, 1.0, 2.0, 4.0],
                            [0.05, 0.15, 0.5, 1.5, 4.0], [0.05, 0.1, 0.2, 0.4, 0.8], 250)
        R[f'convex_tuned_{nm}'] = tuned
        R[f'C1_convex_{nm}'] = study('convex/' + nm, task, lam, 1000, sg, SEEDS, meth, 10)
        json.dump(R, open(OUT, 'w'), indent=1)
        print('convex', nm, time.time() - t0, flush=True)

    # ---------------- non-convex: Fashion-MNIST CNN, 50 devices ------------
    task2 = Task('fmnist_cnn', K=50, seed=0, nper=500, ncls=2, batch=64)
    lam2 = np.geomspace(0.25, 1.0, 50); lam2 /= lam2.mean()
    pil = fl_run(task2, 'prop', 0.1, 40, 0.0, lam2, seed=99, rho=1.0, log_every=1)
    Gmax2 = float(np.array(pil['log']['G']).max() * 1.1)
    R['nonconvex_Gmax'] = Gmax2; R['nonconvex_d'] = task2.d
    for tgt, nm in [(0.25, 'lownoise'), (4.0, 'highnoise')]:
        sg, gb, b0 = pick_sigma(task2, lam2, 0.1, target=tgt)
        R[f'nonconvex_sigma_{nm}'] = dict(sigma=sg, gnorm0=gb, beta0=b0, noise_to_grad=tgt)
        meth, tuned = build(task2, lam2, sg, Gmax2, [0.02, 0.05, 0.1, 0.2],
                            [0.01, 0.05, 0.2, 0.8, 3.0], [0.02, 0.05, 0.1, 0.2], 300)
        R[f'nonconvex_tuned_{nm}'] = tuned
        R[f'C2_nonconvex_{nm}'] = study('nonconvex/' + nm, task2, lam2, 1500, sg, 3, meth, 25)
        json.dump(R, open(OUT, 'w'), indent=1)
        print('nonconvex', nm, time.time() - t0, flush=True)
    print('DONE', time.time() - t0, flush=True)
