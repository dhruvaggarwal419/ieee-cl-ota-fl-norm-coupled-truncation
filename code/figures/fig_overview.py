import json,numpy as np,sys
sys.path.insert(0,'.')
from scipy.special import lambertw
from diagram_kit import *
C=json.load(open('r3_C_convergence.json'))['C1_convex_lownoise']
F=json.load(open('r3_F_causal.json')); M={**F['F1_main'],**F['F1_control']}
rng=np.random.default_rng(2)
D=Diagram(7.16,3.45,100,48)
Y0,YH=10.5,34
D.group(0.8,Y0,15.5,YH,'1. Devices',c='in',fs=6.4)
D.group(17.8,Y0,19.4,YH,'2. Silence test',c='warn',fs=6.4)
D.group(38.8,Y0,34.5,YH,'3. Gain rule',c='stat',fs=6.4)
D.group(74.8,Y0,9.0,YH,'4. Server',c='est',fs=6.2)
D.group(85.6,Y0,13.6,YH,'5. Outcome',c='out',fs=6.2)
# devices thumbnail: gradient norms of 17 majority and 3 minority devices
Gn=np.r_[0.173*np.exp(rng.normal(0,0.18,17)),2.06*np.exp(rng.normal(0,0.12,3))]
ax=D.inset(2.4,19.5,12.4,16.5); cols=['#6C8EBF']*17+['#B85450']*3; ax.bar(range(20),Gn,color=cols,width=0.8); ax.set_xticks([]); ax.set_yticks([0,1,2]); ax.set_title('gradient norm $G_k$',fontsize=5,pad=1)
D.text(8.6,16.8,'3 minority devices\nhave large norms',fs=4.9,color='#B85450',bold=True)
D.text(8.6,12.8,'measured, MNIST,\n$K=20$',fs=4.4)
# silence test
D.box(19.2,32.0,16.6,7.2,'send only if\n$|h_k|^2\\geq a^2G_k^2/P$','white',5.3)
ax=D.inset(19.8,15.6,15.4,13.5); a=np.linspace(0,3.5,200)
for G,c,l in [(0.17,'#6C8EBF','small $G$'),(1.0,'#D79B00','$G=1$'),(2.06,'#B85450','large $G$')]: ax.plot(a,np.exp(-a**2*G**2),color=c,lw=1.0,label=l)
ax.set_title('send probability $p_k(a)$',fontsize=5,pad=1); ax.legend(fontsize=4.1,frameon=False,loc='upper right',handlelength=1.1,borderpad=0.1,labelspacing=0.15); ax.set_xticks([0,2]); ax.set_yticks([0,1])
D.arrow(14.9,34.0,19.1,34.0); D.arrow(21.0,31.9,21.0,29.2)
# gain rule rows
D.box(40.2,29.0,10.6,8.0,'Common gain\n$a_k=\\beta=\\eta$','white',5.4,ec='#A00000')
ax=D.inset(53.2,26.2,18.6,11.6); idx=np.arange(20); pc=np.r_[0.989*np.ones(17),0.666*np.ones(3)]
ax.bar(idx,pc/pc.max(),color=['#6C8EBF']*17+['#B85450']*3,width=0.8); ax.set_ylim(0,1.15); ax.set_xticks([]); ax.set_yticks([0,1]); ax.set_title('expected received gain: minority loses',fontsize=4.8,pad=1)
D.box(40.2,13.5,10.6,8.0,'Norm-aware gain\nLambert W','white',5.4,ec='#106010')
ax=D.inset(53.2,12.3,18.6,11.4)
Gm=np.linspace(0.05,1,120); beta=1/np.sqrt(2*np.e)  # G/Gmax, P*lam=1, rho=1
x=-lambertw(-2*(beta*Gm)**2,0).real/2; ax.plot(Gm,np.exp(-x),color='#2E7D32',lw=1.2); ax.axhline(np.exp(-0.5),color='#B85450',lw=0.7,ls='--'); ax.set_ylim(0.5,1.03)
ax.text(0.5,0.535,'floor $e^{-1/2}=0.61$',fontsize=4.6,color='#B85450',ha='center'); ax.set_xticks([0,1]); ax.set_yticks([0.6,1]); ax.set_title('send probability vs $G_k/G_{\\max}$',fontsize=4.8,pad=1)
D.poly([(35.9,35.6),(38.0,35.6),(38.0,17.5),(40.1,17.5)]); D.arrow(38.0,33.0,40.1,33.0)
D.arrow(50.9,33.0,53.0,33.0); D.arrow(50.9,17.5,53.0,17.5)
D.text(45.5,25.3,'each device\nsets $a_k$ from\nits own norm',fs=4.5)
# server
D.box(75.6,26.0,7.4,9.0,'$\\hat g=\\dfrac{y}{K\\beta}$','white',6.0)
D.text(79.3,22.2,'sum of silent\nand active\nsignals plus\nnoise',fs=4.5)
D.text(79.3,15.0,'needs only\n$\\beta\\leq\\min_k u_k$',fs=4.6)
D.arrow(71.9,31.5,75.5,31.5); D.poly([(71.9,17.5),(74.2,17.5),(74.2,28.5),(75.5,28.5)])
# outcome thumbnails
ax=D.inset(86.9,26.2,11.2,10.4)
for k,l,c,ls in [('proposed','proposed','#2E7D32','-'),('norm_blind_Gmax','fixed bound','#C62828','--')]:
    ax.semilogy(C[k]['rounds'],C[k]['mse_mean'],ls,color=c,lw=1.0,label=l)
ax.set_title('aggregation error',fontsize=5,pad=1); ax.legend(fontsize=4.0,frameon=False,loc='center right',handlelength=1.0,borderpad=0.1); ax.set_xticks([0,1000]); ax.set_yticks([0.01,0.1]); ax.set_yticklabels(['.01','.1'])
ax=D.inset(86.9,14.2,11.2,9.2)
names=[('ideal','Id.'),('common_gain_CONTROL_meannorm','CG*'),('proposed','Pr.'),('common_gain','CG')]
ax.bar(range(4),[M[k]['worst_rec'][0] for k,_ in names],color=['#9E9E9E','#F9A825','#2E7D32','#1565C0'],width=0.7); ax.set_ylim(0.4,0.9); ax.set_xticks(range(4)); ax.set_xticklabels([n for _,n in names],fontsize=4.2); ax.set_yticks([0.5,0.8]); ax.set_title('worst-class recall',fontsize=5,pad=1)
D.arrow(83.1,31.0,86.8,31.0)
D.text(92.4,11.5,'CG*: common gain, norm\nremoved from silence test',fs=3.8)
D.legend([('in','Devices'),('warn','Silence test'),('stat','Gain rule'),('est','Server'),('out','Outcome')],y=5.0,x0=3,dx=19.5)
D.text(50,1.6,'The norm coupling in the silence test explains most of the minority loss. The norm-aware gain is unbiased, and every device sends with probability at least 0.61.',fs=5.2,style='italic')
D.save('../fig_overview.pdf','/tmp/l3v2_ov.png')
