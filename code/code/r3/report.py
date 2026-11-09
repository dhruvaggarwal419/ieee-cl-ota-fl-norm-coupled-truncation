"""Build the plain-text report from the r3 JSON results on the laptop."""
import json, os, numpy as np
D = '/Users/dhruvaggarwal/research/nightly'


def L(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def hdr(t):
    print('\n' + '=' * 78); print(t); print('=' * 78)


def pm(v):
    return '%.4f+-%.4f' % (v[0], v[1]) if isinstance(v, (list, tuple)) else '%.4f' % v


# ---------------------------------------------------------------- A
A = L('r3_A_assumptions.json')
if A:
    hdr('(A) RELAXING THE THEOREM-1 ASSUMPTIONS  (K=20, eta=0.5, sigma=0.05, 5 seeds)')
    print('effect   = ||mean(c_k) - mean-shift prediction||  (size of the bias Thm 1 claims)')
    print('rel_T1   = ||w_sim - Thm1 prediction|| / effect     (theorem AS STATED)')
    print('rel_EX   = ||w_sim - exact E[ghat]=0 point|| / effect  (generalised map)')
    print('err_*    = ||converged point - true optimum of F||\n')
    print('%-26s %8s %8s %8s %9s %9s %9s' %
          ('scenario', 'effect', 'rel_T1', 'rel_EX', 'err_prop', 'err_fixed', 'err_norm'))
    rows = [A['A0_baseline']] + A['A1_curvature'] + A['A2_channel'] + A['A3_both']
    seen = set()
    for r in rows:
        if r['tag'] in seen: continue
        seen.add(r['tag'])
        print('%-26s %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f' %
              (r['tag'], r['effect'], r['rel_T1'], r['rel_EX'],
               r['err_prop'], r['err_fixed'], r['err_norm']))
    for k in ['A4_logistic', 'A5_logistic_hetlam', 'A6_nn']:
        if k not in A: continue
        r = A[k]
        print('\n%s : %s' % (k, r['tag']))
        print('  mean-shift effect size            %.4f' % r['effect'])
        print('  ||w_sim(common gain) - Thm1||     %.4f   (rel %.3f)' % (r['abs_T1'], r['rel_T1']))
        print('  dist to true stationary point:  ideal %.4f  proposed %.4f  common gain %.4f'
              '  normalised %.4f' % (r['dist_ideal_wstar'], r['dist_prop_wstar'],
                                     r['dist_fixed_wstar'], r['dist_norm_wstar']))
        print('  ||grad F|| at converged point:  ideal %.5f  proposed %.5f  common gain %.5f'
              '  normalised %.5f' % (r['ideal']['gradnorm'], r['prop']['gradnorm'],
                                     r['fixed']['gradnorm'], r['norm']['gradnorm']))
        print('  bound (4) (1/K)sum(1-p_k)G_k = %.5f   mean p_k = %.3f' %
              (r['gradnorm_bound'], r['mean_p']))

# ---------------------------------------------------------------- B
B = L('r3_B_imperfect.json')
if B:
    hdr('(B) IMPERFECT GRADIENT-NORM INFORMATION')
    print('B1 synthetic (K=20, d=2, sigma=0.05, 5 seeds).  bias_mean is the relative bias of')
    print('E[ghat] averaged along the trajectory; err is the converged-point error.\n')
    print('%-24s %11s %16s %9s %7s' % ('variant', 'bias_mean', 'conv err', 'beta', 'part'))
    for k, v in B['B1_synth'].items():
        print('%-24s %11.3e %8.4f+-%.4f %9.4f %7.3f' %
              (k, v['bias_mean'], v['err'][0], v['err'][1], v['beta_mean'], v['part']))
    if 'B4_probe_bias' in B:
        print('\nB4 exact relative bias of E[ghat] at three FIXED probe points')
        print('%-24s %12s %12s %12s %10s' % ('variant', 'start', 'halfway', 'near opt', 'capped@start'))
        for k, v in B['B4_probe_bias'].items():
            print('%-24s %12.2e %12.2e %12.2e %10.2f' %
                  (k, v['start w=(1,0)']['rel_bias'], v['halfway']['rel_bias'],
                   v['near optimum']['rel_bias'], v['start w=(1,0)']['frac_devices_capped']))
    if 'B5_noise' in B:
        print('\nB5 converged error vs receiver noise (synthetic)')
        ks = [k for k in B['B5_noise'][0] if k != 'sigma']
        print('%8s ' % 'sigma' + ' '.join('%13s' % k[:13] for k in ks))
        for r in B['B5_noise']:
            print('%8.2f ' % r['sigma'] + ' '.join('%13.4f' % r[k]['err'] for k in ks))
    print('\nB2 learning: MNIST softmax, minority split, sigma=0.05, T=200, 5 seeds')
    print('%-24s %16s %16s %7s %8s %9s' %
          ('variant', 'test acc', 'worst recall', 'part', 'beta', 'agg mse'))
    for k, v in B['B2_mnist'].items():
        print('%-24s %8.4f+-%.4f %8.3f+-%.3f %7.3f %8.4f %9.4g' %
              (k, v['acc'][0], v['acc'][1], v['worst'][0], v['worst'][1],
               v['part'], v['beta'], v['mse']))
    print('\nB3 control-channel overhead, bits per round')
    print('%-22s %6s %4s %6s %10s %12s %18s %10s' %
          ('task', 'd', 'K', 'bits/G', 'control', 'min-only', 'digital payload', 'ratio'))
    for o in B['B3_overhead']:
        if o['bits_per_entry'] != 16: continue
        print('%-22s %6d %4d %6d %10d %12d %18d %10.2e' %
              (o['task'], o['d'], o['K'], o['norm_bits'], o['control_bits_per_round'],
               o['control_bits_minonly'], o['digital_payload_bits_per_round'],
               o['control_over_digital']))

# ---------------------------------------------------------------- C
C = L('r3_C_convergence.json')
if C:
    hdr('(C) EMPIRICAL CONVERGENCE')
    for key in [k for k in C if k.startswith('C1_') or k.startswith('C2_')]:
        sk = key.replace('C1_', '').replace('C2_', '')
        sg = C.get('convex_sigma_' + sk.split('_')[-1]) or C.get('nonconvex_sigma_' + sk.split('_')[-1])
        print('\n--- %s   sigma=%.4g (noise/grad^2 = %s) ---' %
              (key, sg['sigma'] if sg else float('nan'), sg['noise_to_grad'] if sg else '?'))
        print('%-22s %6s %16s %16s %16s %10s %10s %9s' %
              ('method', 'lr', 'final train loss', 'final ||gradF||', 'test acc',
               'rate_loss', 'rate_g2', 'agg mse'))
        for m, v in C[key].items():
            if v.get('diverged'): print('%-22s DIVERGED' % m); continue
            print('%-22s %6g %16s %16s %16s %10s %10s %9.3g' %
                  (m, v['lr'], pm(v['final_loss']), pm(v['final_gnorm']), pm(v['final_acc']),
                   ('%.3f' % v['rate_loss']) if v['rate_loss'] is not None else 'na',
                   ('%.3f' % v['rate_bestgrad2']) if v['rate_bestgrad2'] is not None else 'na',
                   v['agg_mse_mean']))

# ---------------------------------------------------------------- D
Dd = L('r3_D_bottleneck.json')
if Dd:
    hdr('(D) THE MIN-GAIN BOTTLENECK')
    print('D1 scaling in K (heterogeneity 3.0, sigma=0.05, d=2)')
    print('%4s %9s %8s %11s %11s %13s %13s %13s' %
          ('K', 'beta_prop', 'max G', 'var_noise', 'var_trunc', 'err_prop', 'err_fixed', 'err_norm'))
    for r in Dd['D1_K']:
        p = r['w0_prop']
        print('%4d %9.4f %8.3f %11.3g %11.3g %6.4f+-%.4f %6.4f+-%.4f %6.4f+-%.4f' %
              (r['K'], p['beta'], p['Gmax'], p['var_noise'], p['var_trunc'],
               r['converr_prop'][0], r['converr_prop'][1], r['converr_fixed'][0],
               r['converr_fixed'][1], r['converr_norm'][0], r['converr_norm'][1]))
    print('\nD2 scaling in gradient heterogeneity (K=50)')
    print('%6s %9s %8s %11s %13s %13s %13s' %
          ('spread', 'beta_prop', 'max G', 'mse_prop', 'err_prop', 'err_fixed', 'err_norm'))
    for r in Dd['D2_hetero']:
        p = r['w0_prop']
        print('%6.1f %9.4f %8.3f %11.4g %6.4f+-%.4f %6.4f+-%.4f %6.4f+-%.4f' %
              (r['hetero'], p['beta'], p['Gmax'], p['mse'],
               r['converr_prop'][0], r['converr_prop'][1], r['converr_fixed'][0],
               r['converr_fixed'][1], r['converr_norm'][0], r['converr_norm'][1]))
    print('\nD3 alternative gain objective: beta = rho * quantile_q(u_k) (sigma=0.05)')
    print('%4s %5s %9s %11s %11s %11s %14s' %
          ('K', 'q', 'beta', 'bias^2', 'var_trunc', 'mse', 'conv err'))
    for r in Dd['D3_tradeoff']:
        print('%4d %5.2f %9.4f %11.3g %11.3g %11.3g %7.4f+-%.4f' %
              (r['K'], r['q'], r['beta'], r['bias2'], r['var_trunc'], r['mse'],
               r['converr'], r['converr_sd']))
    if 'D5_noise_sweep' in Dd:
        print('\nD5 the same trade-off at higher receiver noise')
        print('%4s %8s %10s %10s %10s %10s %10s' %
              ('K', 'sigma', 'noisefrac', 'err q=0', 'best q', 'err best', 'err fixed'))
        for r in Dd['D5_noise_sweep']:
            c0 = r['curve'][0]; best = min(r['curve'], key=lambda c: c['converr'])
            print('%4d %8.3g %10.2f %10.4f %10.2f %10.4f %10.4f' %
                  (r['K'], r['sigma'], c0['var_noise'] / c0['mse'], c0['converr'],
                   best['q'], best['converr'], r['fixed'][0]))
    if 'D6_beta_growth' in Dd:
        print('\nD6 does beta grow during training?  (paper claims the noise term shrinks with the gradients)')
        for t in Dd['D6_beta_growth']:
            print('  spread %4.1f : beta x%.2f over 4000 rounds, max||g_k|| x%.2f' %
                  (t['hetero'], t['beta_growth_factor'], t['Gmax_shrink_factor']))
    t = Dd['D4_beta_traj']
    print('\nD4 beta of the proposed rule vs the fixed-Gmax rule, K=20')
    print('  round 0    : prop %.4f  blind %.4f  ratio %.2f' %
          (t['beta_prop'][0], t['beta_blind'][0], t['beta_prop'][0] / t['beta_blind'][0]))
    print('  round %4d : prop %.4f  blind %.4f  ratio %.2f' %
          (t['round'][-1], t['beta_prop'][-1], t['beta_blind'][-1],
           t['beta_prop'][-1] / t['beta_blind'][-1]))

# ---------------------------------------------------------------- E
E = L('r3_E_bigexp.json')
if E:
    hdr('(E) CIFAR-10, SMALL CNN, %d DEVICES, SEVERE NON-IID + MINORITY GROUP' % E.get('K', 50))
    print('d = %s parameters, T = %s rounds, %s seeds' % (E.get('d'), E.get('T'), E.get('seeds')))
    for key in [k for k in E if k.startswith('E_')]:
        sg = E.get('sigma_' + key[2:])
        print('\n--- %s  sigma=%.4g ---' % (key, sg['sigma'] if sg else float('nan')))
        print('%-24s %16s %16s %10s %16s %14s %7s' %
              ('method', 'test acc', 'worst recall', 'rec var', 'worst device acc',
               'device acc sd', 'part'))
        for m, v in E[key].items():
            print('%-24s %16s %16s %10.4f %16s %14s %7.3f' %
                  (m, pm(v['acc']), pm(v['worst_rec']), v['rec_var'][0],
                   pm(v['dev_acc_min']), pm(v['dev_acc_std']), v['part']))

# ---------------------------------------------------------------- F
F = L('r3_F_causal.json')
if F:
    hdr('(F) CAUSAL TEST OF THE MINORITY-CLASS MECHANISM')
    for tag in ['F1', 'F2']:
        if tag + '_main' not in F: continue
        print('\n--- %s ---' % ('MNIST softmax, K=20, 3 minority devices, 10 seeds'
                                if tag == 'F1' else 'Fashion-MNIST CNN, K=50, 3 minority devices, 3 seeds'))
        print('%-36s %16s %16s %16s %7s %7s %8s %8s' %
              ('arm', 'test acc', 'worst recall', 'minority recall', 'G_min/G_maj',
               'part', 'p_minor', 'p_major'))
        for blk in [tag + '_main', tag + '_control']:
            if blk not in F: continue
            for m, v in F[blk].items():
                print('%-36s %16s %16s %16s %7.2f %7.3f %8.3f %8.3f' %
                      (m, pm(v['acc']), pm(v['worst_rec']), pm(v['minority_rec']),
                       v['G_ratio'], v['part'], v['p_minority'], v['p_majority']))

# ---------------------------------------------------------------- G
G = L('r3_G_baselines.json')
if G:
    hdr('(G) BASELINES UNDER MATCHED ENERGY AND COMMUNICATION BUDGETS')
    for key, blk in G.items():
        print('\n--- %s   (d=%d, K=%d, T=%d, %d seeds, sigma=%.4g) ---' %
              (blk['tag'], blk['d'], blk['K'], blk['T'], blk['seeds'], blk['sigma']['sigma']))
        print('%-36s %16s %16s %16s %10s %8s' %
              ('method', 'test acc', 'worst recall', 'worst device acc', 'energy', 'part'))
        for m, v in blk['results'].items():
            print('%-36s %16s %16s %16s %10.4f %8.3f' %
                  (m, pm(v['acc']), pm(v['worst_rec']), pm(v['dev_acc_min']),
                   v['energy'], v['part']))
        print('  control bits/round: ' + ', '.join('%s=%s' % (k, v) for k, v in
                                                   blk['control_bits_per_round'].items()))
