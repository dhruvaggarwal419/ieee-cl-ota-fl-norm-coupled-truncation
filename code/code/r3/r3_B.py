"""(B) Imperfect gradient-norm information + control-channel overhead.

Key structural point this experiment tests: the device always knows its own
current G_k exactly (it just computed g_k).  The only thing the CONTROL LINK has
to carry is enough information for the server to set beta <= min_k u_k,
u_k = sqrt(P lam_k/(2e))/G_k.  a_k is then computed locally and a_k p_k = beta
holds exactly.  So an UNDER-estimate of beta costs only noise, while an
OVER-estimate costs bias.  Every variant below is measured on both axes.
"""
import numpy as np, json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from otalib import a_from_beta, a_cheap, gains

P = 1.0
OUT = '/workspace/USER/r3ota/r3_B_imperfect.json'
SIG = 0.05


def quant_log(x, lo, hi, bits, mode='near'):
    x = np.clip(np.asarray(x, float), lo, hi)
    a, b = np.log(lo), np.log(hi); n = 2 ** bits - 1
    z = (np.log(x) - a) / (b - a) * n
    z = {'near': np.round, 'ceil': np.ceil, 'floor': np.floor}[mode](z)
    return np.exp(a + np.clip(z, 0, n) / n * (b - a))


def make(K=20, off=3.0, seed=0):
    r = np.random.default_rng(seed)
    nfar = max(1, K // 5)
    C = np.concatenate([r.normal(0, 0.3, (K - nfar, 2)),
                        np.column_stack([np.full(nfar, off), r.normal(0, 0.3, nfar)])])
    lam = np.geomspace(0.25, 1.0, K); lam /= lam.mean(); r.shuffle(lam)
    return C, lam


def reported(variant, t, G, Ghist, rng, Glo, Ghi):
    kind = variant['kind']
    if kind in ('exact', 'cheap', 'static', 'minonly'): return G
    if kind == 'qG': return quant_log(G, Glo, Ghi, variant['bits'], variant['mode'])
    if kind == 'delay':
        j = max(0, len(Ghist) - 1 - variant['d']); return Ghist[j]
    if kind == 'noisy':
        return np.maximum(G * (1 + variant['eps'] * rng.standard_normal(len(G))), 1e-6)
    raise ValueError(kind)


def run_synth(variant, C, lam, seed=0, T=3000, tail=600, rho=1.0, betastatic=None,
              Glo=1e-3, Ghi=1e3, ulo=1e-4, uhi=1e2):
    K, d = C.shape; r = np.random.default_rng(seed)
    w = np.array([1.0, 0.0]); acc = []; Ghist = []
    bias, betas, parts, viol = [], [], [], 0
    for t in range(T):
        g = w - C; G = np.linalg.norm(g, axis=1); Ghist.append(G)
        u_true = np.sqrt(P * lam / (2 * np.e)) / G
        if variant['kind'] == 'static':
            beta = betastatic
        elif variant['kind'] == 'qu':                      # quantise u_k itself
            beta = rho * quant_log(u_true, ulo, uhi, variant['bits'], variant['mode']).min()
        else:
            Grep = reported(variant, t, G, Ghist, r, Glo, Ghi)
            beta = rho * (np.sqrt(P * lam / (2 * np.e)) / Grep).min()
        b = np.minimum(beta, u_true)
        a = a_cheap(b, G, lam, P, variant.get('iters', 1)) if variant['kind'] == 'cheap' \
            else a_from_beta(b, G, lam, P)
        if (a ** 2 * G ** 2).max() > P * 1.0000001: viol += 1
        p = np.exp(-a ** 2 * G ** 2 / (P * lam))
        eff = a * p
        Eg = (eff[:, None] * g).sum(0) / (K * beta); gb = g.mean(0)
        bias.append(np.linalg.norm(Eg - gb) / max(np.linalg.norm(gb), 1e-12))
        betas.append(beta)
        h2 = r.exponential(lam); sent = h2 >= a ** 2 * G ** 2 / P
        parts.append(sent.mean())
        y = ((sent * a)[:, None] * g).sum(0) + r.normal(0, SIG, d)
        w = w - min(0.5, 1.0 / (1 + t / 200)) * y / (K * beta)
        if not np.isfinite(w).all(): return None
        if t > T - tail: acc.append(w.copy())
    wf = np.mean(acc, 0)
    return dict(w=wf.tolist(), err=float(np.linalg.norm(wf - C.mean(0))),
                bias_mean=float(np.mean(bias)), bias_last=float(np.mean(bias[-tail:])),
                beta_mean=float(np.mean(betas)), beta_last=float(np.mean(betas[-tail:])),
                part=float(np.mean(parts)), power_violations=int(viol))


def variants():
    V = [('exact', dict(kind='exact'))]
    for b in [1, 2, 4, 8]:
        V += [(f'quantG{b}b_nearest', dict(kind='qG', bits=b, mode='near')),
              (f'quantG{b}b_ceil', dict(kind='qG', bits=b, mode='ceil')),
              (f'quantU{b}b_nearest', dict(kind='qu', bits=b, mode='near')),
              (f'quantU{b}b_floor', dict(kind='qu', bits=b, mode='floor'))]
    for d in [1, 2, 5]: V.append((f'delay{d}', dict(kind='delay', d=d)))
    for e in [0.01, 0.05, 0.10, 0.20]: V.append((f'noisy{int(e*100)}pct', dict(kind='noisy', eps=e)))
    V += [('cheapLW_1iter', dict(kind='cheap', iters=1)),
          ('cheapLW_2iter', dict(kind='cheap', iters=2)),
          ('static_beta_noreport', dict(kind='static'))]
    return V


if __name__ == '__main__':
    t0 = time.time(); R = {'sigma': SIG, 'P': P}
    K = 20; C, lam = make(K)
    # dynamic range of G and u over an exact run, used to set the quantiser range
    w = np.array([1.0, 0.0]); Gs = []
    r0 = np.random.default_rng(0)
    for t in range(3000):
        g = w - C; G = np.linalg.norm(g, axis=1); Gs.append(G)
        u = np.sqrt(P * lam / (2 * np.e)) / G; beta = u.min()
        a = a_from_beta(np.minimum(beta, u), G, lam, P)
        h2 = r0.exponential(lam); sent = h2 >= a ** 2 * G ** 2 / P
        w = w - min(0.5, 1.0 / (1 + t / 200)) * (((sent * a)[:, None] * g).sum(0)
                                                 + r0.normal(0, SIG, 2)) / (K * beta)
    Gs = np.array(Gs); Glo, Ghi = float(Gs.min()) / 2, float(Gs.max()) * 2
    us = np.sqrt(P * lam / (2 * np.e))[None, :] / Gs
    ulo, uhi = float(us.min()) / 2, float(us.max()) * 2
    betastatic = float(np.sqrt(P * lam.min() / (2 * np.e)) / Ghi)
    R['ranges'] = dict(Glo=Glo, Ghi=Ghi, ulo=ulo, uhi=uhi, betastatic=betastatic,
                       u_min_dynamic_range=float(us.min(1).max() / us.min(1).min()))
    print('ranges', R['ranges'], time.time() - t0, flush=True)

    R['B1_synth'] = {}
    for name, v in variants():
        out = [run_synth(v, C, lam, seed=s, betastatic=betastatic,
                         Glo=Glo, Ghi=Ghi, ulo=ulo, uhi=uhi) for s in range(5)]
        out = [o for o in out if o]
        R['B1_synth'][name] = dict(
            err=[float(np.mean([o['err'] for o in out])), float(np.std([o['err'] for o in out], ddof=1))],
            bias_mean=float(np.mean([o['bias_mean'] for o in out])),
            bias_last=float(np.mean([o['bias_last'] for o in out])),
            beta_mean=float(np.mean([o['beta_mean'] for o in out])),
            beta_last=float(np.mean([o['beta_last'] for o in out])),
            part=float(np.mean([o['part'] for o in out])),
            power_violations=int(np.sum([o['power_violations'] for o in out])),
            nruns=len(out))
        print('B1', name, R['B1_synth'][name]['bias_mean'], R['B1_synth'][name]['err'][0],
              time.time() - t0, flush=True)
    json.dump(R, open(OUT, 'w'), indent=1)

    # ---------------- B2: learning performance, MNIST softmax, minority split
    from flrun import Task, fl_run
    task = Task('mnist_softmax', K=20, seed=0, nper=595, ncls=7, minority=([7, 8, 9], 3))
    lam20 = np.geomspace(0.25, 1.0, 20); lam20 /= lam20.mean()
    print('d =', task.d, 'devcls', task.devcls, flush=True)

    def mk_gd(v, Glo, Ghi):
        st = {'hist': []}
        def f(t, G, Gh):
            if v['kind'] == 'qG': return quant_log(G, Glo, Ghi, v['bits'], v['mode'])
            if v['kind'] == 'delay': return Gh[max(0, len(Gh) - 1 - v['d'])]
            if v['kind'] == 'noisy':
                return np.maximum(G * (1 + v['eps'] * np.random.default_rng(t + 991).standard_normal(len(G))), 1e-9)
            return G
        return f

    LR = 2.0; T = 200
    # quantiser range from a pilot proposed run
    pil = fl_run(task, 'prop', LR, T, SIG, lam20, seed=0, rho=1.0, log_every=1)
    Gall = np.array(pil['log']['G']); Glo2, Ghi2 = float(Gall.min()) / 2, float(Gall.max()) * 2
    uall = np.sqrt(P * lam20 / (2 * np.e))[None, :] / Gall
    bstat = float(np.sqrt(P * lam20.min() / (2 * np.e)) / Ghi2)
    R['B2_range'] = dict(Glo=Glo2, Ghi=Ghi2, betastatic=bstat, T=T, lr=LR)

    R['B2_mnist'] = {}
    cases = [('exact', dict(rule='prop', kw=dict(rho=1.0)))]
    for b in [1, 2, 4, 8]:
        cases.append((f'quantG{b}b_ceil', dict(rule='prop', kw=dict(rho=1.0),
                                               gd=dict(kind='qG', bits=b, mode='ceil'))))
        cases.append((f'quantG{b}b_nearest', dict(rule='prop', kw=dict(rho=1.0),
                                                  gd=dict(kind='qG', bits=b, mode='near'))))
    for d_ in [1, 2, 5]:
        cases.append((f'delay{d_}', dict(rule='prop', kw=dict(rho=1.0), gd=dict(kind='delay', d=d_))))
    for e in [0.01, 0.05, 0.10, 0.20]:
        cases.append((f'noisy{int(e*100)}pct', dict(rule='prop', kw=dict(rho=1.0),
                                                    gd=dict(kind='noisy', eps=e))))
    cases.append(('cheapLW_1iter', dict(rule='prop', kw=dict(rho=1.0, cheap=True))))
    cases.append(('static_beta_noreport', dict(rule='prop', kw=dict(rho=1.0, betafix=bstat))))
    cases.append(('common_gain_ref', dict(rule='fixed', kw=dict(eta=0.4))))
    for name, cf in cases:
        gd = mk_gd(cf['gd'], Glo2, Ghi2) if 'gd' in cf else None
        rs = [fl_run(task, cf['rule'], LR, T, SIG, lam20, seed=s, gdesign_fn=gd, log_every=50, **cf['kw'])
              for s in range(5)]
        R['B2_mnist'][name] = dict(
            acc=[float(np.mean([x['acc'] for x in rs])), float(np.std([x['acc'] for x in rs], ddof=1))],
            worst=[float(np.mean([x['worst_rec'] for x in rs])), float(np.std([x['worst_rec'] for x in rs], ddof=1))],
            part=float(np.mean([x['part'] for x in rs])),
            beta=float(np.mean([np.mean(x['log']['beta']) for x in rs])),
            mse=float(np.mean([np.mean(x['log']['mse']) for x in rs])))
        print('B2', name, R['B2_mnist'][name]['acc'], R['B2_mnist'][name]['worst'],
              time.time() - t0, flush=True)
        json.dump(R, open(OUT, 'w'), indent=1)

    # ---------------- B3: control-channel overhead accounting --------------
    ov = []
    for tag, d_, K_, bq in [('MNIST softmax (paper)', 1970, 20, 16),
                            ('MNIST softmax (paper)', 1970, 20, 32),
                            ('CIFAR-10 small CNN', 66000, 50, 16),
                            ('CIFAR-10 small CNN', 66000, 50, 32)]:
        for nb in [1, 2, 4, 8, 32]:
            ctrl_up = K_ * nb                 # every device reports its scalar
            ctrl_up_min = nb                  # only min_k u_k is needed (max-consensus)
            ctrl_dn = 32                      # beta broadcast
            dig = K_ * d_ * bq                # orthogonal digital FL payload, bits/round
            ota_uses = d_                     # analog channel uses/round (shared)
            ov.append(dict(task=tag, d=d_, K=K_, bits_per_entry=bq, norm_bits=nb,
                           control_bits_per_round=ctrl_up + ctrl_dn,
                           control_bits_minonly=ctrl_up_min + ctrl_dn,
                           digital_payload_bits_per_round=dig,
                           control_over_digital=(ctrl_up + ctrl_dn) / dig,
                           control_minonly_over_digital=(ctrl_up_min + ctrl_dn) / dig,
                           ota_analog_channel_uses=ota_uses,
                           digital_channel_uses_at_2bps=dig / 2.0,
                           ota_saving_factor=dig / 2.0 / ota_uses))
    R['B3_overhead'] = ov
    json.dump(R, open(OUT, 'w'), indent=1)
    print('DONE', time.time() - t0, flush=True)
