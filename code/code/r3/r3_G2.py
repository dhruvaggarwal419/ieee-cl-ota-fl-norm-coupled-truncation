"""(G, part 2) corrected matched-energy baselines.

The bisection used in r3_G assumed the mean transmit energy is monotone in the
common gain eta.  It is not: as eta -> 0 the energy -> 0 (tiny signals) and as
eta -> infinity the energy -> 0 again (everyone stays silent), so the energy is
unimodal.  Here eta is chosen on a grid, restricted to the rising branch, as the
value whose MEASURED mean energy is closest to the proposed rule's energy.
Replaces the *_MATCHED_ENERGY entries in r3_G_baselines.json.
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, pick_sigma
from r3_G import ci95, summarise

OUT = '/workspace/USER/r3ota/r3_G_baselines.json'


def energy_curve(task, lam, sigma, lr, method, kw, pn, grid, seeds=2, T=150):
    E = []
    for g in grid:
        k = dict(kw); k[pn] = g
        E.append(float(np.mean([fl_run(task, method, lr, T, sigma, lam, seed=800 + s,
                                       log_every=25, **k)['energy'] for s in range(seeds)])))
    return np.array(E)


def pick_eta(grid, E, target):
    pk = int(np.argmax(E))
    lo = E[:pk + 1]; g = grid[:pk + 1]
    if len(g) == 0 or E.max() < target: return float(grid[int(np.argmax(E))]), False
    j = int(np.argmin(np.abs(lo - target)))
    return float(g[j]), True


if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT))
    tasks = [Task('mnist_softmax', K=20, seed=s, nper=595, ncls=7, minority=([7, 8, 9], 3))
             for s in range(10)]
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    grid = list(np.geomspace(0.01, 3.0, 20))
    for key in [k for k in R if k.startswith('G1_') or k.startswith('G2_')]:
        blk = R[key]; sg = blk['sigma']['sigma']; Etar = blk['results']['proposed']['energy']
        Gmean = blk['Gmean']
        blk['energy_curves'] = {}
        for name, m, kw, pn in [('common_gain', 'fixed', {}, 'eta'),
                                ('normalised_update', 'norm', {}, 'eta'),
                                ('gradient_clipping', 'clip', dict(clipC=Gmean), 'eta'),
                                ('channel_aware_sched', 'sched', {}, 'eta')]:
            lr = blk['tuned'][name]['lr']
            E = energy_curve(tasks[0], lam, sg, lr, m, kw, pn, grid)
            eta, ok = pick_eta(np.array(grid), E, Etar)
            k = dict(kw); k[pn] = eta
            runs = [fl_run(tasks[s], m, lr, blk['T'], sg, lam, seed=s, log_every=25, **k)
                    for s in range(blk['seeds'])]
            r = summarise(runs); r.update(lr=lr, par=eta, target_energy=Etar,
                                          reachable=bool(ok))
            blk['results'][name + '_MATCHED_ENERGY'] = r
            blk['energy_curves'][name] = dict(grid=grid, energy=E.tolist())
            print('%s %-22s eta=%.4f E=%.4f (target %.4f) acc=%.4f worst=%.3f  %.0fs' %
                  (key, name, eta, r['energy'], Etar, r['acc'][0], r['worst_rec'][0],
                   time.time() - t0), flush=True)
            json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
