# Letter 3 extension: synthetic checks with CIs over seeds, more clients, heterogeneity, and a stronger baseline (receiver knows participation rate)
import numpy as np, json, time, sys
from scipy.special import lambertw
quick=len(sys.argv)>1
P=1.0; lam=1.0
def make(K,off,spread,seed):
    r=np.random.default_rng(seed)
    A=np.concatenate([r.normal(0,0.3,(K-4,2)),np.array([[off,0.0]]*2+[[off,spread],[off,-spread]])])
    return A
def run(rule,A,eta=None,T=3000,seed=0,sigma=0.05):
    K=len(A); r=np.random.default_rng(seed); w=np.array([1.0,0.0]); tail=[]; pm=[]
    for t in range(T):
        g=w-A; G=np.linalg.norm(g,axis=1); h2=r.exponential(lam,K)
        if rule in('fixed','fixed_knownp'):
            a=np.full(K,eta); part=h2>=a**2*G**2/P; beta=eta
            if rule=='fixed_knownp': beta=eta*max(part.mean(),1.0/K)
        else:
            u=np.sqrt(P*lam/(2*np.e))/G; beta=u.min()
            rr=beta*G/np.sqrt(P*lam); x=-lambertw(-2*rr**2,0).real/2
            a=np.sqrt(x*P*lam)/G; part=h2>=a**2*G**2/P
        y=(a[part,None]*g[part]).sum(0)+r.normal(0,sigma,2)
        step=min(0.5,1.0/(1+t/200)); w=w-step*y/(K*beta)
        if t>T-500: tail.append(w.copy())
    return np.mean(tail,0)
seeds=3 if quick else 10; T=800 if quick else 3000; R={}; t0=time.time()
for K in ([20] if quick else [10,20,50]):
  for off in ([3.0] if quick else [1.5,3.0,6.0]):
    A=make(K,off,0.5,0); mean=A.mean(0); key=f'K{K}_off{off}'; R[key]={'mean':mean.tolist()}
    for name,rule,eta in [('prop','prop',None),('fixed0.3','fixed',0.3),('fixed0.7','fixed',0.7),('fixed_knownp0.3','fixed_knownp',0.3),('fixed_knownp0.7','fixed_knownp',0.7)]:
        e=np.array([run(rule,A,eta,T,s)-mean for s in range(seeds)])
        n=np.linalg.norm(e,axis=1); R[key][name]={'err_mean':float(n.mean()),'err_std':float(n.std(ddof=1)) if seeds>1 else 0.0}
    json.dump(R,open('ota_ext.json','w'),indent=1); print(key,time.time()-t0,flush=True)
print('done',flush=True)
