"""Control: is the normalised-update baseline bad because of the CHANNEL, or is
normalised aggregation simply a bad estimator under this heterogeneity?
Runs the normalised and clipped updates with a PERFECT channel (no truncation,
no receiver noise) and compares with ideal FedAvg.  Appends I to r3_G_baselines.json."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, tune, pick_sigma

OUT = '/workspace/USER/r3ota/r3_G_baselines.json'
S = 10

def ci95(x):
    from scipy.stats import t as tdist
    x = np.asarray(x, float); n = len(x)
    return [float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))]

if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tasks = [Task('mnist_softmax', K=20, seed=s, nper=595, ncls=7, minority=([7, 8, 9], 3))
             for s in range(S)]
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    pil = fl_run(tasks[0], 'prop', 2.0, 60, 0.0, lam, seed=99, rho=1.0, log_every=1)
    Gmean = float(np.array(pil['log']['G']).mean())
    out = {'note': 'perfect channel: no truncation, no receiver noise', 'seeds': S,
           'Gmean': Gmean}
    for name, m, lrs, kw in [('ideal_fedavg', 'ideal', [0.5, 1, 2, 4], {}),
                             ('normalised_PERFECT_channel', 'norm_ideal',
                              [0.05, 0.15, 0.5, 1.5, 4.0], {}),
                             ('clipped_PERFECT_channel', 'clip_ideal',
                              [0.5, 1, 2, 4, 8], dict(clipC=Gmean))]:
        v, lr, p = tune(tasks[0], m, lam, 200, 0.0, lrs, (None,), None, **kw)
        runs = [fl_run(tasks[s], m, lr, 200, 0.0, lam, seed=s, log_every=50, **kw)
                for s in range(S)]
        rec = np.array([r['rec'] for r in runs])
        out[name] = dict(lr=lr, acc=ci95([r['acc'] for r in runs]),
                         worst=ci95([r['worst_rec'] for r in runs]),
                         minority_rec=ci95(rec[:, [7, 8, 9]].mean(1)))
        print(name, 'lr=%g acc=%.4f+-%.4f worst=%.3f+-%.3f' %
              (lr, out[name]['acc'][0], out[name]['acc'][1], out[name]['worst'][0],
               out[name]['worst'][1]), flush=True)
    R['I_perfect_channel_controls'] = out
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
