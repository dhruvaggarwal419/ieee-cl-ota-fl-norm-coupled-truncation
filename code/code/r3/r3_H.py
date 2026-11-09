"""Direct check of the paper's headline claim, with enough seeds to settle it.

Paper (Min split, sigma=0.05, 10 seeds): worst-class recall
  proposed 0.729 +- 0.036, common gain 0.633 +- 0.061, norm-blind 0.558 +- 0.040;
  test accuracy 87.7 +- 0.5, 87.4 +- 0.6, 78.9 +- 0.7 percent.

Here: 20 seeds, each with its own device split and its own channels/noise, the
same quantities, plus the PAIRED difference (proposed minus common gain) with a
95% CI and a sign test, which is what decides whether the gap is real.
Appends H to r3_F_causal.json.
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, tune, pick_sigma

OUT = '/workspace/USER/r3ota/r3_F_causal.json'
S = 20


def ci95(x):
    from scipy.stats import t as tdist
    x = np.asarray(x, float); n = len(x)
    return [float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))]


if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    from scipy.stats import ttest_rel, binomtest
    tasks = [Task('mnist_softmax', K=20, seed=s, nper=595, ncls=7, minority=([7, 8, 9], 3))
             for s in range(S)]
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    pil = fl_run(tasks[0], 'prop', 2.0, 60, 0.0, lam, seed=99, rho=1.0, log_every=1)
    Gmax = float(np.array(pil['log']['G']).max() * 1.1)
    sgc, gb, b0 = pick_sigma(tasks[0], lam, 2.0, target=4.0)
    out = {'seeds': S, 'Gmax': Gmax}
    for signame, sg in [('paper_sigma_0.05', 0.05), ('calibrated_highnoise', sgc)]:
        blk = {'sigma': sg}
        spec = [('proposed', 'prop', [0.5, 1, 2, 4], (None,), None, dict(rho=1.0)),
                ('common_gain', 'fixed', [0.5, 1, 2, 4], [0.05, 0.1, 0.2, 0.4, 0.8], 'eta', {}),
                ('norm_blind_Gmax', 'blind', [0.5, 1, 2, 4], (None,), None,
                 dict(rho=1.0, Gmax=Gmax)),
                ('ideal', 'ideal', [0.5, 1, 2, 4], (None,), None, {})]
        per = {}
        for name, m, lrs, pars, pn, kw in spec:
            v, lr, p = tune(tasks[0], m, lam, 200, sg, lrs, pars, pn, **kw)
            k = dict(kw)
            if pn is not None and p is not None: k[pn] = p
            runs = [fl_run(tasks[s], m, lr, 200, sg, lam, seed=s, log_every=50, **k)
                    for s in range(S)]
            per[name] = dict(worst=[r['worst_rec'] for r in runs],
                             acc=[r['acc'] for r in runs],
                             minrec=[float(np.mean(np.array(r['rec'])[[7, 8, 9]])) for r in runs])
            blk[name] = dict(lr=lr, par=p, worst=ci95(per[name]['worst']),
                             acc=ci95(per[name]['acc']), minority_rec=ci95(per[name]['minrec']))
            print(signame, name, 'acc=%.4f+-%.4f worst=%.3f+-%.3f' %
                  (blk[name]['acc'][0], blk[name]['acc'][1], blk[name]['worst'][0],
                   blk[name]['worst'][1]), flush=True)
        for metric in ['worst', 'acc', 'minrec']:
            a = np.array(per['proposed'][metric]); b = np.array(per['common_gain'][metric])
            d = a - b
            t, pv = ttest_rel(a, b)
            w = int((d > 0).sum()); l = int((d < 0).sum())
            sg_ = binomtest(w, w + l, 0.5).pvalue if (w + l) > 0 else 1.0
            blk[f'paired_{metric}_prop_minus_common'] = dict(
                mean_diff=float(d.mean()), ci=ci95(d)[1], t=float(t), p_value=float(pv),
                wins=w, losses=l, sign_test_p=float(sg_))
            print('  paired %-7s diff=%+.4f +-%.4f  p=%.4f  wins %d/%d' %
                  (metric, d.mean(), ci95(d)[1], pv, w, w + l), flush=True)
        blk['per_seed'] = per
        out[signame] = blk
        R['H_headline_claim'] = out
        json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
