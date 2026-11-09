import json,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'serif','font.size':7})
S=json.load(open('../results/synth.json'))
def load(sc,s): return json.load(open(f'../results/{sc}_{s}.json'))[s]
def ms(runs,key):
    x=np.array([r[key] for r in runs]); return x.mean(),stats.t.ppf(0.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
fig,ax=plt.subplots(1,2,figsize=(3.5,1.9))
e=S['etas']; ax[0].plot(e,[t[0] for t in S['fixed_theory']],'-',c='#B85450',label='Theorem 1')
ax[0].errorbar(e,[f[0] for f in S['fixed_sim']],[f[1] for f in S['fixed_sim']],fmt='o',ms=2.5,c='#6C8EBF',label='Common gain, sim.')
ax[0].axhline(S['mean'][0],c='#82B366',ls='--',label='True average')
ax[0].plot([0.5],[S['prop_sim'][0]],'s',c='#82B366',ms=3,label='Proposed (sim.)')
ax[0].set_xlabel('Common gain $\\eta$'); ax[0].set_ylabel('Fixed point (axis 1)'); ax[0].legend(fontsize=5,frameon=False,loc='center right'); ax[0].set_title('(a)',fontsize=7)
sig=['0.0','0.01','0.05']; sv=[float(s) for s in sig]
col={'ideal':'k','fixed':'#6C8EBF','blind':'#D6B656','prop':'#82B366'}; nm={'ideal':'Ideal','fixed':'Common gain','blind':'Norm-blind','prop':'Proposed'}
R={s:load('minor',s) for s in sig}
for m in ['ideal','fixed','blind','prop']:
    y=[ms(R[s][m]['runs'],'minrec') for s in sig]
    ax[1].errorbar(np.array(sv)+({'ideal':0,'fixed':-0.0008,'blind':0.0008,'prop':0.0016}[m]),[a for a,b in y],[b for a,b in y],fmt='-o',ms=2.5,lw=0.8,c=col[m],label=nm[m],capsize=1.5)
ax[1].set_xlabel('Noise std $\\sigma$'); ax[1].set_ylabel('Worst-class recall'); ax[1].legend(fontsize=5,frameon=False,loc='lower left'); ax[1].set_title('(b)',fontsize=7)
fig.tight_layout(pad=0.4); fig.savefig('../fig_results.pdf')
