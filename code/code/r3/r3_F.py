"""(F) Causal evidence for the minority-class gain.

Logs per round: per-device gradient norm, per-device transmission probability,
realised participation count, and per-class recall.

Control experiment: the gradient-norm dependence is removed from the TRUNCATION
TEST ONLY (the silence test uses a constant norm instead of the device's own
norm).  Everything else - payload, gains, divisor, channel draws, seed - is
identical.  Two controls are run: (a) the plain mean-norm control, and (b) the
same control with the constant scaled so that the MEAN participation matches the
uncontrolled run, which removes the confound of a different silence rate.
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, tune, pick_sigma

OUT = '/workspace/USER/r3ota/r3_F_causal.json'


def ci95(x):
    from scipy.stats import t as tdist
    x = np.asarray(x, float); n = len(x)
    if n < 2: return [float(x.mean()), 0.0]
    return [float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))]


def diag(runs, minor_idx):
    """gradient-norm / participation diagnostics, minority vs majority devices."""
    G = np.array([r['log']['G'] for r in runs])        # (S, R, K)
    pd = np.array([r['log']['pdes'] for r in runs])
    sn = np.array([r['log']['sent'] for r in runs]).astype(float)
    maj = np.array([k for k in range(G.shape[2]) if k not in minor_idx])
    mi = np.array(minor_idx)
    return dict(G_minority=float(G[:, :, mi].mean()), G_majority=float(G[:, :, maj].mean()),
                G_ratio=float(G[:, :, mi].mean() / G[:, :, maj].mean()),
                p_minority=float(pd[:, :, mi].mean()), p_majority=float(pd[:, :, maj].mean()),
                sent_minority=float(sn[:, :, mi].mean()), sent_majority=float(sn[:, :, maj].mean()),
                sent_count_mean=float(sn.sum(2).mean()),
                G_minority_curve=G[:, :, mi].mean((0, 2)).tolist(),
                G_majority_curve=G[:, :, maj].mean((0, 2)).tolist(),
                p_minority_curve=pd[:, :, mi].mean((0, 2)).tolist(),
                p_majority_curve=pd[:, :, maj].mean((0, 2)).tolist(),
                sent_count_curve=sn.sum(2).mean(0).tolist(),
                sent_minority_curve=sn[:, :, mi].mean((0, 2)).tolist(),
                sent_majority_curve=sn[:, :, maj].mean((0, 2)).tolist(),
                recall_rounds=[r[0] for r in runs[0]['log']['rec']],
                recall_curve=np.array([[x[1] for x in r['log']['rec']]
                                       for r in runs]).mean(0).tolist(),
                rounds=runs[0]['log']['round'])


def block(task, lam, sigma, lr, methods, seeds, T, minor_idx, log_every, tag, minor_cls):
    out = {}
    for name, m, kw in methods:
        runs = [fl_run(task, m, lr, T, sigma, lam, seed=s, log_every=log_every, **kw)
                for s in range(seeds)]
        rec = np.array([r['rec'] for r in runs])
        out[name] = dict(acc=ci95([r['acc'] for r in runs]),
                         worst_rec=ci95([r['worst_rec'] for r in runs]),
                         minority_rec=ci95(rec[:, minor_cls].mean(1)),
                         rec_per_class=rec.mean(0).tolist(),
                         part=float(np.mean([r['part'] for r in runs])),
                         **diag(runs, minor_idx))
        print(tag, name, 'acc=%.4f worst=%.3f minrec=%.3f part=%.3f  Gratio=%.2f pmin=%.3f pmaj=%.3f'
              % (out[name]['acc'][0], out[name]['worst_rec'][0], out[name]['minority_rec'][0],
                 out[name]['part'], out[name]['G_ratio'], out[name]['p_minority'],
                 out[name]['p_majority']), flush=True)
    return out


def calibrate(task, lam, sigma, lr, method, kw, target_part, seeds=2, T=200):
    """scale the constant test norm so the mean participation matches target_part."""
    lo, hi = 0.2, 5.0
    for _ in range(12):
        mid = np.sqrt(lo * hi)
        p = np.mean([fl_run(task, method, lr, T, sigma, lam, seed=700 + s, log_every=10,
                            fix_test_norm=mid, **kw)['part'] for s in range(seeds)])
        if p > target_part: lo = mid          # larger test norm -> less participation
        else: hi = mid
    return float(np.sqrt(lo * hi))


if __name__ == '__main__':
    t0 = time.time(); R = {}

    # =============== F1: the paper's own setup (MNIST, 20 devices, 3 minority)
    task = Task('mnist_softmax', K=20, seed=0, nper=595, ncls=7, minority=([7, 8, 9], 3))
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    minor_idx = [17, 18, 19]
    R['F1_minor_idx'] = minor_idx; R['F1_devcls'] = task.devcls
    sg, gb, b0 = pick_sigma(task, lam, 2.0, target=4.0)
    R['F1_sigma'] = dict(sigma=sg, gnorm0=gb, beta0=b0)
    ve, lre, etae = tune(task, 'fixed', lam, 250, sg, [0.5, 1.0, 2.0, 4.0],
                         [0.05, 0.1, 0.2, 0.4, 0.8], 'eta')
    vp, lrp, _ = tune(task, 'prop', lam, 250, sg, [0.5, 1.0, 2.0, 4.0], (None,), None, rho=1.0)
    R['F1_tuned'] = dict(fixed=[ve, lre, etae], prop=[vp, lrp])
    LR = lrp
    base = [('proposed', 'prop', dict(rho=1.0)),
            ('common_gain', 'fixed', dict(eta=etae)),
            ('ideal', 'ideal', {})]
    R['F1_main'] = block(task, lam, sg, LR, base, 10, 200, minor_idx, 10, 'F1', [7, 8, 9])
    # controls: remove the gradient-norm dependence from the silence test only
    pf = R['F1_main']['common_gain']['part']; pp = R['F1_main']['proposed']['part']
    sf = calibrate(task, lam, sg, LR, 'fixed', dict(eta=etae), pf)
    spz = calibrate(task, lam, sg, LR, 'prop', dict(rho=1.0), pp)
    R['F1_calib'] = dict(fixed_scale=sf, prop_scale=spz, target_fixed=pf, target_prop=pp)
    ctrl = [('common_gain_CONTROL_meannorm', 'fixed', dict(eta=etae, fix_test_norm=True)),
            ('common_gain_CONTROL_matchedpart', 'fixed', dict(eta=etae, fix_test_norm=sf)),
            ('proposed_CONTROL_meannorm', 'prop', dict(rho=1.0, fix_test_norm=True)),
            ('proposed_CONTROL_matchedpart', 'prop', dict(rho=1.0, fix_test_norm=spz))]
    R['F1_control'] = block(task, lam, sg, LR, ctrl, 10, 200, minor_idx, 10, 'F1ctrl', [7, 8, 9])
    json.dump(R, open(OUT, 'w'), indent=1)
    print('F1 done', time.time() - t0, flush=True)

    # =============== F2: same question at scale (Fashion-MNIST CNN, 50 devices)
    K = 50
    task2 = Task('fmnist_cnn', K=K, seed=0, nper=500, ncls=2, minority=([8, 9], 3), batch=64)
    lam2 = np.geomspace(0.25, 1.0, K); lam2 /= lam2.mean()
    minor2 = [K - 3, K - 2, K - 1]
    sg2, gb2, b2 = pick_sigma(task2, lam2, 0.1, target=4.0)
    R['F2_sigma'] = dict(sigma=sg2, gnorm0=gb2, beta0=b2)
    ve, lre2, etae2 = tune(task2, 'fixed', lam2, 300, sg2, [0.02, 0.05, 0.1, 0.2],
                           [0.02, 0.05, 0.1, 0.2], 'eta')
    vp, lrp2, _ = tune(task2, 'prop', lam2, 300, sg2, [0.02, 0.05, 0.1, 0.2], (None,), None, rho=1.0)
    R['F2_tuned'] = dict(fixed=[ve, lre2, etae2], prop=[vp, lrp2])
    base2 = [('proposed', 'prop', dict(rho=1.0)),
             ('common_gain', 'fixed', dict(eta=etae2)),
             ('ideal', 'ideal', {})]
    R['F2_main'] = block(task2, lam2, sg2, lrp2, base2, 3, 1200, minor2, 25, 'F2', [8, 9])
    json.dump(R, open(OUT, 'w'), indent=1)
    pf2 = R['F2_main']['common_gain']['part']; pp2 = R['F2_main']['proposed']['part']
    sf2 = calibrate(task2, lam2, sg2, lrp2, 'fixed', dict(eta=etae2), pf2, seeds=1, T=150)
    sp2 = calibrate(task2, lam2, sg2, lrp2, 'prop', dict(rho=1.0), pp2, seeds=1, T=150)
    R['F2_calib'] = dict(fixed_scale=sf2, prop_scale=sp2)
    ctrl2 = [('common_gain_CONTROL_meannorm', 'fixed', dict(eta=etae2, fix_test_norm=True)),
             ('common_gain_CONTROL_matchedpart', 'fixed', dict(eta=etae2, fix_test_norm=sf2)),
             ('proposed_CONTROL_matchedpart', 'prop', dict(rho=1.0, fix_test_norm=sp2))]
    R['F2_control'] = block(task2, lam2, sg2, lrp2, ctrl2, 3, 1200, minor2, 25, 'F2ctrl', [8, 9])
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
