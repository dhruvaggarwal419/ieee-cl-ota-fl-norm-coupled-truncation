import json,numpy as np,glob,sys
from scipy import stats
import os
sig=[x for x in ['0.0','0.01','0.02','0.05'] if os.path.exists(f'../results/mnist_{x}.json')]; meths=['ideal','fixed','blind','prop']
import sys
SC=sys.argv[1]
sig=[x for x in ['0.0','0.01','0.02','0.05'] if os.path.exists(f'../results/{SC}_{x}.json')]
R={s:json.load(open(f'../results/{SC}_{s}.json'))[s] for s in sig}
def ms(runs,key):
    x=np.array([r[key] for r in runs]); h=stats.t.ppf(0.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); return x.mean(),h
for s in sig:
    for m in meths:
        d=R[s][m]; print(s,m,'lr',d['lr'],'par',d['par'],' '.join(f"{k}={ms(d['runs'],k)[0]:.4f}+-{ms(d['runs'],k)[1]:.4f}" for k in ['test','minrec','loss','minloc','part']), 'div',sum(r['diverged'] for r in d['runs']))
