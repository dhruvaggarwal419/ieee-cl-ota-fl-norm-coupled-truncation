"""(G) Baselines under matched energy and communication budgets.

Communication: every analog method uses exactly d channel uses per round; they
differ only in control-channel bits, which are counted explicitly.
Energy: besides the tuned version of each baseline, a second version is run whose
free gain parameter is calibrated by bisection so that the MEASURED mean transmit
energy per device-round equals that of the proposed rule.
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, tune, pick_sigma

OUT = '/workspace/USER/r3ota/r3_G_baselines.json'


def ci95(x):
    from scipy.stats import t as tdist
    x = np.asarray(x, float); n = len(x)
    if n < 2: return [float(x.mean()), 0.0]
    return [float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))]


def summarise(runs):
    rec = np.array([r['rec'] for r in runs])
    return dict(acc=ci95([r['acc'] for r in runs]),
                worst_rec=ci95([r['worst_rec'] for r in runs]),
                rec_var=ci95([r['rec_var'] for r in runs]),
                dev_acc_min=ci95([r['dev_acc_min'] for r in runs]),
                dev_acc_std=ci95([r['dev_acc_std'] for r in runs]),
                part=float(np.mean([r['part'] for r in runs])),
                energy=float(np.mean([r['energy'] for r in runs])),
                beta=float(np.mean([np.mean(r['log']['beta']) for r in runs])),
                agg_mse=float(np.mean([np.mean(r['log']['mse']) for r in runs])),
                diverged=int(sum(r['diverged'] for r in runs)))


def match_energy(task, lam, sigma, lr, method, kw, parname, target_E, T=200, seeds=2,
                 lo=0.005, hi=3.0):
    for _ in range(14):
        mid = np.sqrt(lo * hi); k = dict(kw); k[parname] = mid
        E = np.mean([fl_run(task, method, lr, T, sigma, lam, seed=800 + s, log_every=10, **k)['energy']
                     for s in range(seeds)])
        if E > target_E: hi = mid
        else: lo = mid
    return float(np.sqrt(lo * hi))


def run_task(tag, tasks, lam, lrs_m, lrs_u, etas, T, seeds, sigma_target, Ttune):
    task = tasks[0]          # tuning and calibration use split 0 only
    out = {'tag': tag, 'd': task.d, 'K': task.K, 'T': T, 'seeds': seeds,
           'note': 'each seed uses its own device split AND its own channels/noise'}
    sg, gb, b0 = pick_sigma(task, lam, lrs_m[len(lrs_m) // 2], target=sigma_target)
    out['sigma'] = dict(sigma=sg, gnorm0=gb, beta0=b0, noise_to_grad=sigma_target)
    pil = fl_run(task, 'prop', lrs_m[len(lrs_m) // 2], min(60, T), 0.0, lam, seed=99,
                 rho=1.0, log_every=1)
    Gmax = float(np.array(pil['log']['G']).max() * 1.1)
    Gmean = float(np.array(pil['log']['G']).mean())
    out['Gmax'] = Gmax; out['Gmean'] = Gmean
    spec = [('ideal_fedavg', 'ideal', lrs_m, (None,), None, {}),
            ('common_gain', 'fixed', lrs_m, etas, 'eta', {}),
            ('proposed', 'prop', lrs_m, (None,), None, dict(rho=1.0)),
            ('abrar_michelusi_blind_Gmax', 'blind', lrs_m, (None,), None, dict(rho=1.0, Gmax=Gmax)),
            ('normalised_update', 'norm', lrs_u, etas, 'eta', {}),
            ('normalised_unbiased', 'norm_ub', lrs_u, (None,), None, dict(rho=1.0)),
            ('gradient_clipping', 'clip', lrs_m, etas, 'eta', dict(clipC=Gmean)),
            ('channel_aware_sched', 'sched', lrs_m, etas, 'eta', {}),
            ('qfedavg_ideal', 'qfed_ideal', lrs_m, (None,), None, dict(qfed_q=1.0)),
            ('qfedavg_ota', 'cfixed', lrs_m, etas, 'eta', dict(qfed_q=1.0))]
    res = {}; tuned = {}
    for name, m, lrs, pars, pn, kw in spec:
        v, lr, p = tune(task, m, lam, Ttune, sg, lrs, pars, pn, **kw)
        k = dict(kw)
        if pn is not None and p is not None: k[pn] = p
        runs = [fl_run(tasks[s], m, lr, T, sg, lam, seed=s, log_every=25, **k)
                for s in range(seeds)]
        res[name] = summarise(runs); res[name].update(lr=lr, par=p, tuneloss=v)
        print(tag, name, 'acc=%.4f worst=%.3f E=%.4f part=%.3f' %
              (res[name]['acc'][0], res[name]['worst_rec'][0], res[name]['energy'],
               res[name]['part']), flush=True)
        tuned[name] = dict(lr=lr, par=p)
    out['tuned'] = tuned
    # ---- matched-energy versions of the gain-parameter baselines -----------
    Etar = res['proposed']['energy']
    out['target_energy'] = Etar
    for name, m, kw, pn in [('common_gain', 'fixed', {}, 'eta'),
                            ('normalised_update', 'norm', {}, 'eta'),
                            ('gradient_clipping', 'clip', dict(clipC=Gmean), 'eta'),
                            ('channel_aware_sched', 'sched', {}, 'eta')]:
        lr = tuned[name]['lr']
        e = match_energy(task, lam, sg, lr, m, kw, pn, Etar, T=min(200, T))
        k = dict(kw); k[pn] = e
        runs = [fl_run(tasks[s], m, lr, T, sg, lam, seed=s, log_every=25, **k)
                for s in range(seeds)]
        res[name + '_MATCHED_ENERGY'] = summarise(runs)
        res[name + '_MATCHED_ENERGY'].update(lr=lr, par=e)
        print(tag, name + '_matchedE', 'eta=%.4f acc=%.4f worst=%.3f E=%.4f' %
              (e, res[name + '_MATCHED_ENERGY']['acc'][0],
               res[name + '_MATCHED_ENERGY']['worst_rec'][0],
               res[name + '_MATCHED_ENERGY']['energy']), flush=True)
    out['results'] = res
    # ---- control-channel cost of each method, bits per round --------------
    K = task.K
    out['control_bits_per_round'] = {
        'ideal_fedavg': 'n/a (no wireless model)',
        'common_gain': 0, 'channel_aware_sched': K,            # 1 activity bit per device
        'normalised_update': 0, 'abrar_michelusi_blind_Gmax': 0,
        'normalised_unbiased': 0,
        'gradient_clipping': 0,
        'qfedavg_ota': K * 16,                                  # loss scalar per device
        'proposed_full_reports': K * 8 + 32,                    # 8-bit norm per device + beta
        'proposed_min_only': 8 + 32}                            # only min_k u_k is needed
    return out


if __name__ == '__main__':
    t0 = time.time(); R = {}
    tasks = [Task('mnist_softmax', K=20, seed=s, nper=595, ncls=7, minority=([7, 8, 9], 3))
             for s in range(10)]
    lam = np.geomspace(0.25, 1.0, 20); lam /= lam.mean()
    R['G1_mnist_minority_highnoise'] = run_task(
        'mnist-min/high', tasks, lam, [0.5, 1.0, 2.0, 4.0], [0.05, 0.15, 0.5, 1.5, 4.0],
        [0.05, 0.1, 0.2, 0.4, 0.8], 200, 10, 4.0, 250)
    json.dump(R, open(OUT, 'w'), indent=1); print('G1 done', time.time() - t0, flush=True)
    R['G2_mnist_minority_lownoise'] = run_task(
        'mnist-min/low', tasks, lam, [0.5, 1.0, 2.0, 4.0], [0.05, 0.15, 0.5, 1.5, 4.0],
        [0.05, 0.1, 0.2, 0.4, 0.8], 200, 10, 0.25, 250)
    json.dump(R, open(OUT, 'w'), indent=1); print('G2 done', time.time() - t0, flush=True)

    # the CNN-scale baseline table lives in r3_E_bigexp.json (same method list),
    # so it is not repeated here.
    R['G3_note'] = 'CNN-scale baselines are in r3_E_bigexp.json (Fashion-MNIST and CIFAR-10)'
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
