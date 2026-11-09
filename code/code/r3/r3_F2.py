"""(F, part 2) the same causal test at scale: Fashion-MNIST CNN, 50 devices,
3 minority devices holding classes 8 and 9.  Appends F2 to r3_F_causal.json.
Run separately so the finished F1 results are never overwritten."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, tune, pick_sigma
from r3_F import block, calibrate

OUT = '/workspace/USER/r3ota/r3_F_causal.json'

if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    K = 50
    task2 = Task('fmnist_cnn', K=K, seed=0, nper=500, ncls=2, minority=([8, 9], 3), batch=64)
    lam2 = np.geomspace(0.25, 1.0, K); lam2 /= lam2.mean()
    minor2 = [K - 3, K - 2, K - 1]
    sg2, gb2, b2 = pick_sigma(task2, lam2, 0.1, target=4.0)
    R['F2_sigma'] = dict(sigma=sg2, gnorm0=gb2, beta0=b2)
    print('sigma', sg2, time.time() - t0, flush=True)
    ve, lre2, etae2 = tune(task2, 'fixed', lam2, 150, sg2, [0.05, 0.1], [0.05, 0.1, 0.2], 'eta')
    print('tuned fixed', lre2, etae2, time.time() - t0, flush=True)
    vp, lrp2, _ = tune(task2, 'prop', lam2, 150, sg2, [0.05, 0.1], (None,), None, rho=1.0)
    print('tuned prop', lrp2, time.time() - t0, flush=True)
    R['F2_tuned'] = dict(fixed=[ve, lre2, etae2], prop=[vp, lrp2])
    base2 = [('proposed', 'prop', dict(rho=1.0)),
             ('common_gain', 'fixed', dict(eta=etae2)),
             ('ideal', 'ideal', {})]
    R['F2_main'] = block(task2, lam2, sg2, lrp2, base2, 3, 800, minor2, 25, 'F2', [8, 9])
    json.dump(R, open(OUT, 'w'), indent=1)
    pf2 = R['F2_main']['common_gain']['part']; pp2 = R['F2_main']['proposed']['part']
    sf2 = calibrate(task2, lam2, sg2, lrp2, 'fixed', dict(eta=etae2), pf2, seeds=1, T=100)
    sp2 = calibrate(task2, lam2, sg2, lrp2, 'prop', dict(rho=1.0), pp2, seeds=1, T=100)
    R['F2_calib'] = dict(fixed_scale=sf2, prop_scale=sp2, target_fixed=pf2, target_prop=pp2)
    print('calib', sf2, sp2, time.time() - t0, flush=True)
    ctrl2 = [('common_gain_CONTROL_meannorm', 'fixed', dict(eta=etae2, fix_test_norm=True)),
             ('common_gain_CONTROL_matchedpart', 'fixed', dict(eta=etae2, fix_test_norm=sf2)),
             ('proposed_CONTROL_matchedpart', 'prop', dict(rho=1.0, fix_test_norm=sp2))]
    R['F2_control'] = block(task2, lam2, sg2, lrp2, ctrl2, 3, 800, minor2, 25, 'F2ctrl', [8, 9])
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
