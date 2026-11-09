"""(D, part 2) Where does the min-gain bottleneck actually dominate?
Sweep the receiver-noise level (equivalently d*sigma^2) and, at each level, trace
the bias/variance trade-off curve of the quantile-ceiling gain rule.
Appends D5/D6 to r3_D_bottleneck.json."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otalib import gains, step_ota
from r3_D import make, mse_terms, sim

P = 1.0
OUT = '/workspace/USER/r3ota/r3_D_bottleneck.json'

if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    SEEDS = 5
    R['D5_noise_sweep'] = []
    for K in [20, 100]:
        C, lam = make(K, 3.0); wstar = C.mean(0)
        w0 = np.array([1.0, 0.0]); g = w0 - C; G = np.linalg.norm(g, axis=1)
        for sig in [0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
            row = {'K': K, 'sigma': sig, 'curve': []}
            for q in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
                m = mse_terms(g, G, lam, 'quant', sig, 2, rho=1.0, q=q)
                e = [np.linalg.norm(sim('quant', C, lam, seed=s, sigma=sig, rho=1.0, q=q) - wstar)
                     for s in range(SEEDS)]
                m.update(q=q, converr=float(np.mean(e)), converr_sd=float(np.std(e, ddof=1)))
                row['curve'].append(m)
            # reference rules at the same noise level
            for nm, kw in [('fixed', dict(rule='fixed', eta=0.5)),
                           ('norm', dict(rule='norm', eta=0.5)),
                           ('blind', dict(rule='blind', rho=1.0, Gmax=float(G.max() * 1.1)))]:
                e = [np.linalg.norm(sim(kw['rule'], C, lam, seed=s, sigma=sig,
                                        **{k: v for k, v in kw.items() if k != 'rule'}) - wstar)
                     for s in range(SEEDS)]
                row[nm] = [float(np.mean(e)), float(np.std(e, ddof=1))]
                row[nm + '_mse'] = mse_terms(g, G, lam, kw['rule'], sig, 2,
                                             **{k: v for k, v in kw.items() if k != 'rule'})
            R['D5_noise_sweep'].append(row)
            best = min(row['curve'], key=lambda c: c['converr'])
            print('D5 K=%d sigma=%.3g best q=%.2f converr=%.4f (q=0 -> %.4f) noisefrac(q=0)=%.2f %.0fs'
                  % (K, sig, best['q'], best['converr'], row['curve'][0]['converr'],
                     row['curve'][0]['var_noise'] / row['curve'][0]['mse'], time.time() - t0), flush=True)
        json.dump(R, open(OUT, 'w'), indent=1)

    # ---- D6: does beta really grow as training proceeds?  (max-norm trajectory)
    R['D6_beta_growth'] = []
    for het in [1.0, 3.0, 8.0]:
        K = 20; C, lam = make(K, het)
        r = np.random.default_rng(0); w = np.array([1.0, 0.0])
        G0 = np.linalg.norm(w - C, axis=1)
        tr = {'hetero': het, 'round': [], 'Gmax': [], 'Gmean': [], 'beta': []}
        for t in range(4000):
            g = w - C; G = np.linalg.norm(g, axis=1)
            gh, sent, a, beta, en = step_ota(r, g, G, lam, P, 'prop', 0.05, 2, rho=1.0)
            if t % 100 == 0:
                tr['round'].append(t); tr['Gmax'].append(float(G.max()))
                tr['Gmean'].append(float(G.mean())); tr['beta'].append(float(beta))
            w = w - min(0.5, 1.0 / (1 + t / 200)) * gh
        tr['beta_growth_factor'] = tr['beta'][-1] / tr['beta'][0]
        tr['Gmax_shrink_factor'] = tr['Gmax'][-1] / tr['Gmax'][0]
        R['D6_beta_growth'].append(tr)
        print('D6 het=%.1f beta x%.2f  Gmax x%.2f' % (het, tr['beta_growth_factor'],
                                                      tr['Gmax_shrink_factor']), flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
