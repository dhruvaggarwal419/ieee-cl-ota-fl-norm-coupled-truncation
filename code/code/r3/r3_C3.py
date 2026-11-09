"""(C, part 3) the gain beta over training: does it really grow as the gradients
shrink, and how much larger is it than the fixed-Gmax (norm-blind) design?
Appends C3 to r3_C_convergence.json."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, pick_sigma

OUT = '/workspace/USER/r3ota/r3_C_convergence.json'

if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    task = Task('mnist_softmax', K=20, seed=0, nper=595, ncls=7, minority=([7, 8, 9], 3))
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    sg, gb, b0 = pick_sigma(task, lam, 2.0, target=4.0)
    pil = fl_run(task, 'prop', 4.0, 60, 0.0, lam, seed=99, rho=1.0, log_every=1)
    Gmax = float(np.array(pil['log']['G']).max() * 1.1)
    out = {'sigma': sg, 'Gmax': Gmax, 'lr': 4.0, 'T': 1000, 'seeds': 3}
    for name, m, kw in [('proposed', 'prop', dict(rho=1.0)),
                        ('norm_blind_Gmax', 'blind', dict(rho=1.0, Gmax=Gmax))]:
        runs = [fl_run(task, m, 4.0, 1000, sg, lam, seed=s, log_every=10, **kw)
                for s in range(3)]
        b = np.array([r['log']['beta'] for r in runs]).mean(0)
        g = np.array([r['log']['G'] for r in runs])          # (S, R, K)
        mse = np.array([r['log']['mse'] for r in runs]).mean(0)
        out[name] = dict(rounds=runs[0]['log']['round'], beta=b.tolist(),
                         Gmax_curve=g.max(2).mean(0).tolist(),
                         Gmean_curve=g.mean((0, 2)).tolist(), mse=mse.tolist(),
                         beta_first=float(b[0]), beta_last=float(b[-1]),
                         beta_growth=float(b[-1] / b[0]),
                         mse_first=float(mse[0]), mse_last=float(mse[-1]),
                         mse_drop=float(mse[0] / max(mse[-1], 1e-12)),
                         acc=float(np.mean([r['acc'] for r in runs])))
        print(name, 'beta %.4f -> %.4f (x%.2f)  aggMSE %.4g -> %.4g (/%.1f)  acc %.4f' %
              (out[name]['beta_first'], out[name]['beta_last'], out[name]['beta_growth'],
               out[name]['mse_first'], out[name]['mse_last'], out[name]['mse_drop'],
               out[name]['acc']), flush=True)
    out['beta_ratio_prop_over_blind_first'] = out['proposed']['beta_first'] / out['norm_blind_Gmax']['beta_first']
    out['beta_ratio_prop_over_blind_last'] = out['proposed']['beta_last'] / out['norm_blind_Gmax']['beta_last']
    R['C3_beta_trajectory'] = out
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
