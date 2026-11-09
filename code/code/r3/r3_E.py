"""(E) Real learning experiment: CIFAR-10, small CNN, 50 devices, severe non-IID
(2 classes per majority device) plus a minority group (classes 8,9 held by only
3 devices).  Multiple seeds with 95% CIs.  Reports test accuracy, worst-class
recall, per-class recall variance and between-device accuracy disparity."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flrun import Task, fl_run, pick_sigma, tune

OUT = '/workspace/USER/r3ota/r3_E_bigexp.json'
K = 50; T = 500; SEEDS = 3; TTUNE = 80


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
                rec_per_class=rec.mean(0).tolist(),
                minority_rec=ci95(rec[:, 8:].mean(1)),
                dev_acc_mean=ci95([r['dev_acc_mean'] for r in runs]),
                dev_acc_min=ci95([r['dev_acc_min'] for r in runs]),
                dev_acc_std=ci95([r['dev_acc_std'] for r in runs]),
                part=float(np.mean([r['part'] for r in runs])),
                energy=float(np.mean([r['energy'] for r in runs])),
                beta=float(np.mean([np.mean(r['log']['beta']) for r in runs])),
                agg_mse=float(np.mean([np.mean(r['log']['mse']) for r in runs])),
                diverged=int(sum(r['diverged'] for r in runs)))


def run_dataset(dsname, label, R, t0, levels=None, subset=None):
    tasks = [Task(dsname, K=K, seed=s, nper=500, ncls=2, minority=([8, 9], 3), batch=64)
             for s in range(SEEDS)]
    task = tasks[0]                       # tuning uses split 0 only
    out = {'dataset': label, 'd': task.d, 'devcls': task.devcls, 'K': K, 'T': T,
           'seeds': SEEDS,
           'note': 'each seed uses its own device split AND its own channels/noise'}
    lam = np.geomspace(0.25, 1.0, K); lam /= lam.mean()
    print(label, 'd', task.d, 'data', tuple(task.Xd.shape), time.time() - t0, flush=True)
    pil = fl_run(task, 'prop', 0.1, 50, 0.0, lam, seed=99, rho=1.0, log_every=1)
    Gmax = float(np.array(pil['log']['G']).max() * 1.1); out['Gmax'] = Gmax
    Gmean = float(np.array(pil['log']['G']).mean()); out['Gmean_pilot'] = Gmean
    KEY = ['ideal_fedavg', 'common_gain', 'proposed', 'norm_blind_Gmax', 'normalised']
    for tgt, nm in (levels or [(4.0, 'highnoise'), (0.25, 'lownoise')]):
        sg, gb, b0 = pick_sigma(task, lam, 0.1, target=tgt)
        out[f'sigma_{nm}'] = dict(sigma=sg, gnorm0=gb, beta0=b0, noise_to_grad=tgt)
        lrs_m = [0.05, 0.1]; lrs_u = [0.05, 0.5]
        etas = [0.05, 0.1, 0.2]
        spec = [('ideal_fedavg', 'ideal', lrs_m, (None,), None, {}),
                ('common_gain', 'fixed', lrs_m, etas, 'eta', {}),
                ('proposed', 'prop', lrs_m, (None,), None, dict(rho=1.0)),
                ('norm_blind_Gmax', 'blind', lrs_m, (None,), None, dict(rho=1.0, Gmax=Gmax)),
                ('normalised', 'norm', lrs_u, etas, 'eta', {}),
                ('normalised_unbiased', 'norm_ub', lrs_u, (None,), None, dict(rho=1.0)),
                ('clipped', 'clip', lrs_m, etas, 'eta', dict(clipC=Gmean)),
                ('sched_known_active', 'sched', lrs_m, etas, 'eta', {}),
                ('qfedavg_ideal', 'qfed_ideal', lrs_m, (None,), None, dict(qfed_q=1.0)),
                ('qfedavg_ota', 'cfixed', lrs_m, etas, 'eta', dict(qfed_q=1.0))]
        if nm == 'lownoise': spec = [x for x in spec if x[0] in KEY]
        if subset is not None: spec = [x for x in spec if x[0] in subset]
        res = {}; tuned = {}
        for name, m, lrs, pars, pn, kw in spec:
            print(label, nm, 'tuning', name, '%.0fs' % (time.time() - t0), flush=True)
            v, lr, p = tune(task, m, lam, TTUNE, sg, lrs, pars, pn, **kw)
            print(label, nm, 'running', name, 'lr=%g par=%s' % (lr, p),
                  '%.0fs' % (time.time() - t0), flush=True)
            k = dict(kw)
            if pn is not None and p is not None: k[pn] = p
            tuned[name] = dict(tuneloss=v, lr=lr, par=p)
            runs = [fl_run(tasks[s], m, lr, T, sg, lam, seed=s, log_every=50, **k)
                    for s in range(SEEDS)]
            res[name] = summarise(runs); res[name]['lr'] = lr; res[name]['par'] = p
            print(label, nm, name, 'acc=%.4f+-%.3f worst=%.3f devmin=%.3f' %
                  (res[name]['acc'][0], res[name]['acc'][1], res[name]['worst_rec'][0],
                   res[name]['dev_acc_min'][0]), '%.0fs' % (time.time() - t0), flush=True)
            out[f'E_{nm}'] = res; out[f'tuned_{nm}'] = tuned
            R[label] = out
            json.dump(R, open(OUT, 'w'), indent=1)
    return out


if __name__ == '__main__':
    t0 = time.time(); R = {}
    SUB = ['ideal_fedavg', 'common_gain', 'proposed', 'norm_blind_Gmax',
           'normalised', 'clipped', 'qfedavg_ota']
    run_dataset('fmnist_cnn', 'Fashion-MNIST', R, t0, levels=[(4.0, 'highnoise')], subset=SUB)
    print('FMNIST DONE', time.time() - t0, flush=True)
    for _ in range(360):                      # wait for the CIFAR download
        if os.path.isdir('/workspace/USER/r3ota/data/cifar-10-batches-py'): break
        time.sleep(30)
    if os.path.isdir('/workspace/USER/r3ota/data/cifar-10-batches-py'):
        run_dataset('cifar_cnn', 'CIFAR-10', R, t0, levels=[(4.0, 'highnoise')],
                    subset=SUB)
    else:
        R['cifar_note'] = 'CIFAR-10 download did not complete in time'
        json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
