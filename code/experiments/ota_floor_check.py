# Analysis of the norm-aware unbiased gain.
# Setup: device k sends a_k g_k, participates iff h_k^2 >= a_k^2||g_k||^2/P, h_k^2~Exp(lam).
#   p_k = exp(-a_k^2||g_k||^2/(P lam)).  Unbiasedness requires a_k p_k = beta for all k.
# Put x_k = a_k^2||g_k||^2/(P lam) and r_k = beta||g_k||/sqrt(P lam). Then r_k = sqrt(x_k)e^{-x_k},
#   so r_k^2 = x_k e^{-2x_k}  =>  x_k = -W_0(-2 r_k^2)/2.   Feasible iff 2r_k^2 <= 1/e.
# Feasibility for every k forces beta <= u_min, u_k = sqrt(P lam/(2e))/||g_k||.
# CLAIM A: with beta = rho*u_min, rho<=1, we get x_k <= 1/2 hence p_k >= e^{-1/2} for EVERY k.
import numpy as np
from scipy.special import lambertw
P=1.0; lam=1.0
def x_of(r): return -lambertw(-2*r*r,0).real/2
rng=np.random.default_rng(0)
print("CLAIM A check: minimum participation over random gradient-norm profiles")
worst=1.0
for trial in range(2000):
    K=rng.integers(3,200)
    G=np.exp(rng.normal(0,rng.uniform(0.1,3.0),K))      # heavy heterogeneity
    rho=rng.uniform(0.05,1.0)
    beta=rho*np.sqrt(P*lam/(2*np.e))/G.max()
    r=beta*G/np.sqrt(P*lam)
    assert (2*r**2).max()<=1/np.e+1e-12, "infeasible"
    x=x_of(r); p=np.exp(-x)
    worst=min(worst,p.min())
print(f"  smallest participation probability seen over 2000 random profiles: {worst:.6f}")
print(f"  predicted floor exp(-1/2) = {np.exp(-0.5):.6f}")
print(f"  holds: {worst >= np.exp(-0.5)-1e-9}")

print("\nCLAIM B: variance of the unbiased aggregate")
# y = sum_k 1{part} a_k g_k + n,  update uses y/(K beta)
# Var = [ sum_k a_k^2||g_k||^2 p_k(1-p_k) ] /(K beta)^2 + d sigma^2/(K beta)^2
# with a_k = beta/p_k  =>  first term = (1/K^2) sum_k ||g_k||^2 (1-p_k)/p_k
# and (1-p)/p <= e^{1/2}-1 since p >= e^{-1/2}
print(f"  bound on (1-p)/p  = e^(1/2)-1 = {np.exp(0.5)-1:.6f}")
# empirical check by simulation
d=2
for K,spread in [(20,1.0),(50,2.0),(100,0.5)]:
    G=np.exp(rng.normal(0,spread,K)); gdir=rng.normal(0,1,(K,d)); g=gdir/np.linalg.norm(gdir,axis=1,keepdims=True)*G[:,None]
    beta=np.sqrt(P*lam/(2*np.e))/G.max()
    r=beta*G/np.sqrt(P*lam); x=x_of(r); p=np.exp(-x); a=np.sqrt(x*P*lam)/G
    assert np.allclose(a*p,beta), (a*p)[:3]
    sig=0.05
    N=200000; acc=np.zeros((N,d))
    h2=rng.exponential(lam,(N,K)); part=h2>=(a**2*G**2/P)[None,:]
    y=np.einsum('nk,kd->nd',part*a[None,:],g)+rng.normal(0,sig,(N,d))
    upd=y/(K*beta)
    emp_mean=upd.mean(0); true=g.mean(0)
    emp_var=((upd-upd.mean(0))**2).sum(1).mean()
    pred_var=(G**2*(1-p)/p).sum()/K**2 + d*sig**2/(K*beta)**2
    bound=(np.exp(0.5)-1)*(G**2).sum()/K**2 + d*sig**2*2*np.e*G.max()**2/(K**2*P*lam)
    print(f"  K={K:4d} spread={spread}: bias={np.linalg.norm(emp_mean-true):.5f}  "
          f"var emp={emp_var:.5f} pred={pred_var:.5f} bound={bound:.5f}  "
          f"part rate min={part.mean(0).min():.4f}")
