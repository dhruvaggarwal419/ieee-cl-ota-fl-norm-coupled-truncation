"""(C, part 2) the non-convex convergence block, run separately so the convex
results are never at risk.  Appends to r3_C_convergence.json."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, pick_sigma, tune
from r3_C import study, build, OUT

if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    task2 = Task('fmnist_cnn', K=50, seed=0, nper=500, ncls=2, batch=64)
    lam2 = np.geomspace(0.25, 1.0, 50); lam2 /= lam2.mean()
    pil = fl_run(task2, 'prop', 0.1, 40, 0.0, lam2, seed=99, rho=1.0, log_every=1)
    Gmax2 = float(np.array(pil['log']['G']).max() * 1.1)
    R['nonconvex_Gmax'] = Gmax2; R['nonconvex_d'] = task2.d
    for tgt, nm in [(4.0, 'highnoise')]:
        sg, gb, b0 = pick_sigma(task2, lam2, 0.1, target=tgt)
        R[f'nonconvex_sigma_{nm}'] = dict(sigma=sg, gnorm0=gb, beta0=b0, noise_to_grad=tgt)
        meth, tuned = build(task2, lam2, sg, Gmax2, [0.02, 0.05, 0.1, 0.2],
                            [0.01, 0.05, 0.2, 0.8, 3.0], [0.02, 0.05, 0.1, 0.2], 200)
        R[f'nonconvex_tuned_{nm}'] = tuned
        R[f'C2_nonconvex_{nm}'] = study('nonconvex/' + nm, task2, lam2, 1000, sg, 3, meth, 25)
        json.dump(R, open(OUT, 'w'), indent=1)
        print('nonconvex', nm, time.time() - t0, flush=True)
    print('DONE', time.time() - t0, flush=True)
