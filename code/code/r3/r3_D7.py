"""(D, part 3) The min-gain bottleneck in a real learning task.
MNIST softmax, minority split, total training data held FIXED while K grows, so
the only thing that changes is how many devices share the gain.
Appends D7 to r3_D_bottleneck.json."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, tune, pick_sigma

OUT = '/workspace/USER/r3ota/r3_D_bottleneck.json'
TOTAL = 12000


def ci95(x):
    from scipy.stats import t as tdist
    x = np.asarray(x, float); n = len(x)
    if n < 2: return [float(x.mean()), 0.0]
    return [float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))]


if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R['D7_K_learning'] = []
    SEEDS = 5
    for K in [5, 10, 20, 50, 100]:
        nmin = max(1, int(round(0.15 * K)))
        nper = max(21, (TOTAL // K // 7) * 7)
        tasks = [Task('mnist_softmax', K=K, seed=s, nper=nper, ncls=7,
                      minority=([7, 8, 9], nmin)) for s in range(SEEDS)]
        lam = np.geomspace(0.25, 1.0, K); lam /= lam.mean()
        sg, gb, b0 = pick_sigma(tasks[0], lam, 2.0, target=4.0)
        pil = fl_run(tasks[0], 'prop', 2.0, 60, 0.0, lam, seed=99, rho=1.0, log_every=1)
        Gmax = float(np.array(pil['log']['G']).max() * 1.1)
        row = {'K': K, 'nper': nper, 'n_minority_dev': nmin, 'sigma': sg, 'Gmax': Gmax,
               'gnorm0': gb, 'beta0': b0}
        for name, m, lrs, pars, pn, kw in [
                ('ideal', 'ideal', [0.5, 1, 2, 4], (None,), None, {}),
                ('common_gain', 'fixed', [0.5, 1, 2, 4], [0.05, 0.1, 0.2, 0.4, 0.8], 'eta', {}),
                ('proposed', 'prop', [0.5, 1, 2, 4], (None,), None, dict(rho=1.0)),
                ('blind_Gmax', 'blind', [0.5, 1, 2, 4], (None,), None, dict(rho=1.0, Gmax=Gmax)),
                ('quantile10', 'quant', [0.5, 1, 2, 4], (None,), None, dict(rho=1.0, q=0.1)),
                ('normalised', 'norm', [0.05, 0.15, 0.5, 1.5, 4.0],
                 [0.05, 0.1, 0.2, 0.4, 0.8], 'eta', {})]:
            v, lr, p = tune(tasks[0], m, lam, 200, sg, lrs, pars, pn, **kw)
            k = dict(kw)
            if pn is not None and p is not None: k[pn] = p
            runs = [fl_run(tasks[s], m, lr, 200, sg, lam, seed=s, log_every=10, **k)
                    for s in range(SEEDS)]
            row[name] = dict(acc=ci95([r['acc'] for r in runs]),
                             worst=ci95([r['worst_rec'] for r in runs]),
                             beta=float(np.mean([np.mean(r['log']['beta']) for r in runs])),
                             agg_mse=float(np.mean([np.mean(r['log']['mse']) for r in runs])),
                             part=float(np.mean([r['part'] for r in runs])), lr=lr, par=p)
            print('D7 K=%3d %-12s acc=%.4f worst=%.3f beta=%.4f aggmse=%.4g  %.0fs' %
                  (K, name, row[name]['acc'][0], row[name]['worst'][0], row[name]['beta'],
                   row[name]['agg_mse'], time.time() - t0), flush=True)
        R['D7_K_learning'].append(row)
        json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
