"""(B, part 2) Two things B1 could not answer cleanly:
 1. the residual bias measured at FIXED probe points (the trajectory-averaged
    relative bias is meaningless near the optimum, where ||grad F|| -> 0);
 2. the noise regime.  In the 2-D synthetic the receiver-noise term is tiny, so a
    conservative (too small) beta looks free.  Sweep sigma so the trade-off shows.
Appends B4/B5 to r3_B_imperfect.json."""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otalib import a_from_beta, a_cheap
from r3_B import quant_log, make, run_synth, variants

P = 1.0
OUT = '/workspace/USER/r3ota/r3_B_imperfect.json'


def probe_bias(variant, C, lam, w, Glo, Ghi, ulo, uhi, betastatic, rho=1.0, seed=0):
    """exact bias of E[ghat] at a fixed w (no trajectory averaging)."""
    K = len(C); r = np.random.default_rng(seed)
    g = w - C; G = np.linalg.norm(g, axis=1)
    u_true = np.sqrt(P * lam / (2 * np.e)) / G
    kind = variant['kind']
    if kind == 'static': beta = betastatic
    elif kind == 'qu':
        beta = rho * quant_log(u_true, ulo, uhi, variant['bits'], variant['mode']).min()
    elif kind == 'qG':
        beta = rho * (np.sqrt(P * lam / (2 * np.e))
                      / quant_log(G, Glo, Ghi, variant['bits'], variant['mode'])).min()
    elif kind == 'noisy':
        Gr = np.maximum(G * (1 + variant['eps'] * r.standard_normal(K)), 1e-6)
        beta = rho * (np.sqrt(P * lam / (2 * np.e)) / Gr).min()
    elif kind == 'delay':
        # a stale norm from d rounds back: emulate with a 10% per-round drift
        Gr = G * (1.0 + 0.1) ** variant['d']
        beta = rho * (np.sqrt(P * lam / (2 * np.e)) / Gr).min()
    else: beta = rho * u_true.min()
    b = np.minimum(beta, u_true)
    a = a_cheap(b, G, lam, P, variant.get('iters', 1)) if kind == 'cheap' \
        else a_from_beta(b, G, lam, P)
    p = np.exp(-a ** 2 * G ** 2 / (P * lam))
    Eg = ((a * p)[:, None] * g).sum(0) / (K * beta); gb = g.mean(0)
    return dict(rel_bias=float(np.linalg.norm(Eg - gb) / np.linalg.norm(gb)),
                beta=float(beta), gradnorm=float(np.linalg.norm(gb)),
                frac_devices_capped=float(np.mean(u_true < beta - 1e-15)))


if __name__ == '__main__':
    t0 = time.time()
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rg = R['ranges']; Glo, Ghi, ulo, uhi = rg['Glo'], rg['Ghi'], rg['ulo'], rg['uhi']
    bstat = rg['betastatic']
    K = 20; C, lam = make(K)
    wmean = C.mean(0)

    # ---- B4: bias at fixed probe points -----------------------------------
    probes = {'start w=(1,0)': np.array([1.0, 0.0]),
              'halfway': 0.5 * (np.array([1.0, 0.0]) + wmean),
              'near optimum': wmean + np.array([0.05, 0.0])}
    R['B4_probe_bias'] = {}
    for name, v in variants():
        R['B4_probe_bias'][name] = {pn: probe_bias(v, C, lam, pw, Glo, Ghi, ulo, uhi, bstat)
                                    for pn, pw in probes.items()}
    for name in R['B4_probe_bias']:
        d = R['B4_probe_bias'][name]
        print('%-24s' % name, ' '.join('%s rel=%.3e beta=%.4f' % (k[:5], d[k]['rel_bias'], d[k]['beta'])
                                       for k in probes), flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)

    # ---- B5: noise sweep of the synthetic converged error ------------------
    import r3_B
    R['B5_noise'] = []
    keep = ['exact', 'quantG1b_ceil', 'quantG2b_ceil', 'quantG4b_ceil', 'quantG4b_nearest',
            'quantG8b_ceil', 'delay5', 'noisy20pct', 'cheapLW_1iter', 'static_beta_noreport']
    vd = dict(variants())
    for sig in [0.05, 0.5, 2.0, 5.0]:
        r3_B.SIG = sig
        row = {'sigma': sig}
        for name in keep:
            out = [run_synth(vd[name], C, lam, seed=s, betastatic=bstat,
                             Glo=Glo, Ghi=Ghi, ulo=ulo, uhi=uhi) for s in range(5)]
            out = [o for o in out if o]
            row[name] = dict(err=float(np.mean([o['err'] for o in out])),
                             err_sd=float(np.std([o['err'] for o in out], ddof=1)),
                             beta=float(np.mean([o['beta_mean'] for o in out])),
                             part=float(np.mean([o['part'] for o in out])))
        R['B5_noise'].append(row)
        print('B5 sigma=%.2f ' % sig,
              ' '.join('%s=%.4f' % (n[:10], row[n]['err']) for n in keep), flush=True)
        json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
