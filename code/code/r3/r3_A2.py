"""(A, part 2) non-quadratic losses: logistic (convex) and a small tanh net
(non-convex).  Split out of r3_A.py so the quadratic results are never lost."""
import numpy as np, json, sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_A as A

OUT = '/workspace/USER/r3ota/r3_A_assumptions.json'

if __name__ == '__main__':
    torch.set_num_threads(4)
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    K = 20
    X, Y, th = A.make_data(K=K, n=200, d=4, off=3.0, seed=0)
    print('data ready', tuple(X.shape), time.time() - t0, flush=True)

    lam1 = np.ones(K)
    R['A4_logistic'] = A.run_model('logistic d=4, equal lambda', A.Logi(4), X, Y, lam1, T=3000)
    json.dump(R, open(OUT, 'w'), indent=1); print('A4 done', time.time() - t0, flush=True)

    lamh = np.geomspace(0.25, 4.0, K); lamh /= lamh.mean()
    R['A5_logistic_hetlam'] = A.run_model('logistic d=4, 16x lambda spread', A.Logi(4), X, Y,
                                          lamh, T=3000)
    json.dump(R, open(OUT, 'w'), indent=1); print('A5 done', time.time() - t0, flush=True)

    nn_model = A.Net(4, 8)
    w0 = np.random.default_rng(7).normal(0, 0.5, nn_model.np)
    R['A6_nn'] = A.run_model('tanh net h=8 (non-convex)', nn_model, X, Y, lam1, T=3000, w0=w0)
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
