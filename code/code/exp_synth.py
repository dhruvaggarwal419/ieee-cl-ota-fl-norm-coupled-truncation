# Synthetic check of Theorem 1 (mean-shift) and Theorem 2 (norm-aware unbiased rule).
import numpy as np, json
from scipy.special import lambertw
rng=np.random.default_rng(0)
K=20; P=1.0; lam=1.0
A=np.concatenate([rng.normal(0,0.3,(16,2)), np.array([[3.0,0.0]]*2+[[3.0,0.5],[3.0,-0.5]])])
mean=A.mean(0)
def expected_fixed(eta, w0):
    w=w0.copy()
    for _ in range(20000):
        g=w-A; p=np.exp(-eta**2*(g**2).sum(1)/(P*lam))
        w=w-0.5*(p[:,None]*g).sum(0)/K
    return w
def sim(rule, eta=None, T=6000, seed=0, sigma=0.0, rho=1.0):
    r=np.random.default_rng(seed); w=np.array([1.0,0.0]); tail=[]
    for t in range(T):
        g=w-A; G=np.linalg.norm(g,axis=1)
        h2=r.exponential(lam,K)
        if rule=='fixed':
            a=np.full(K,eta); part=h2>=a**2*G**2/P; beta=eta
        else:
            u=np.sqrt(P*lam/(2*np.e))/G; beta=rho*u.min()
            rr=beta*G/np.sqrt(P*lam); x=-lambertw(-2*rr**2,0).real/2
            a=np.sqrt(x*P*lam)/G; part=h2>=a**2*G**2/P
        y=(a[part,None]*g[part]).sum(0)+r.normal(0,sigma,2)
        step=min(0.5,1.0/(1+t/200))  # decaying step
        w=w-step*y/(K*beta)
        if t>T-1000: tail.append(w.copy())
    return np.mean(tail,0)
out={'mean':mean.tolist(),'etas':[],'fixed_theory':[],'fixed_sim':[],'prop_sim':[]}
for eta in [0.05,0.1,0.2,0.3,0.4,0.5,0.7,1.0]:
    out['etas'].append(eta)
    th=expected_fixed(eta,np.array([1.0,0.0]))
    out['fixed_theory'].append(th.tolist())
    # note: fixed-eta sim divides by K*eta (receiver does not know p)
    s=[sim('fixed',eta,seed=s_)[0] for s_ in range(5)]
    out['fixed_sim'].append([float(np.mean(s)),float(np.std(s))])
p=[sim('prop',seed=s_,sigma=0.05)[0] for s_ in range(5)]
out['prop_sim']=[float(np.mean(p)),float(np.std(p))]
# KDE mode check for eta=0.5: fixed point equation
json.dump(out,open('/workspace/USER/comm-letter-ota/synth.json','w'),indent=1)
print(json.dumps(out,indent=1))
