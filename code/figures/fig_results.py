import json,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='serif'; plt.rcParams['font.size']=7.5
C=json.load(open('r3_C_convergence.json'))['C1_convex_lownoise']
F=json.load(open('r3_F_causal.json'))
fig,ax=plt.subplots(1,2,figsize=(3.5,2.35),gridspec_kw=dict(width_ratios=[1.0,1.0]))
a=ax[0]
sty=[('proposed','Proposed','#2E7D32','-'),('norm_blind_Gmax','Fixed bound','#C62828','--'),('common_gain','Common gain','#1565C0',':')]
for k,l,c,ls in sty:
    r=np.array(C[k]['rounds']); m=np.array(C[k]['mse_mean'])
    a.semilogy(r,m,ls,color=c,lw=1.3,label=l)
a.set_xlabel('Round'); a.set_ylabel('Aggregation error'); a.legend(fontsize=6,frameon=False,loc='center right'); a.grid(alpha=0.25,lw=0.4)
a.set_title('(a)',fontsize=8)
b=ax[1]
M={**F['F1_main'],**F['F1_control']}
names=[('ideal','Ideal'),('common_gain_CONTROL_meannorm','CG no norm'),('proposed','Prop.'),('common_gain','CG')]
vals=[M[k]['worst_rec'][0] for k,_ in names]; errs=[M[k]['worst_rec'][1] for k,_ in names]
cols=['#9E9E9E','#F9A825','#2E7D32','#1565C0']
b.bar(range(4),vals,yerr=errs,color=cols,capsize=2,width=0.65,ecolor='#333',error_kw=dict(lw=0.7))
b.set_xticks(range(4)); b.set_xticklabels([n for _,n in names],fontsize=6,rotation=30,ha='right'); b.set_ylim(0.4,0.9)
b.set_ylabel('Worst-class recall'); b.grid(alpha=0.25,lw=0.4,axis='y'); b.set_title('(b)',fontsize=8)
fig.tight_layout(pad=0.4,w_pad=0.8)
fig.savefig('../fig_results.pdf',bbox_inches='tight',pad_inches=0.02)
print({n:(round(v,4),round(e,4)) for (k,n),v,e in zip(names,vals,errs)})
