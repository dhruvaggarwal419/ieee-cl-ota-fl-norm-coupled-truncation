import numpy as np, torch, json, sys, time, os, urllib.request, gzip
from scipy.special import lambertw
try:
    _t=torch.zeros(1,device='cuda'); dev='cuda'
except Exception:
    dev='cpu'; torch.set_num_threads(int(os.environ.get('NT','24')))
print('device',dev,flush=True)
K=20; P=1.0; T=200; D='/workspace/USER/comm-letter-ota/data'
os.makedirs(D,exist_ok=True)
def load():
    base='https://ossci-datasets.s3.amazonaws.com/mnist/'
    def get(f):
        p=f'{D}/{f}'
        if not os.path.exists(p): urllib.request.urlretrieve(base+f,p)
        return gzip.open(p).read()
    xi=np.frombuffer(get('train-images-idx3-ubyte.gz'),np.uint8,offset=16).reshape(-1,784)/255.
    yi=np.frombuffer(get('train-labels-idx1-ubyte.gz'),np.uint8,offset=8)
    xt=np.frombuffer(get('t10k-images-idx3-ubyte.gz'),np.uint8,offset=16).reshape(-1,784)/255.
    yt=np.frombuffer(get('t10k-labels-idx1-ubyte.gz'),np.uint8,offset=8)
    pool=lambda x: x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196)
    f=lambda x: np.hstack([pool(x.reshape(-1,28,28).reshape(-1,28*28)) if False else pool(x),np.ones((len(x),1))]).astype(np.float32)
    return f(xi),yi,f(xt),yt
Xtr,Ytr,Xte,Yte=load()
Xte_t=torch.tensor(Xte,device=dev); Yte_t=torch.tensor(Yte,device=dev)
SCEN=os.environ.get('SCEN','std')
def split(seed):
    r=np.random.default_rng(seed); idx=r.permutation(60000); val=idx[:6000]; tr=idx[6000:]
    if SCEN=='minor':
        maj=tr[Ytr[tr]<7][:17*600].reshape(17,600)
        mino=np.stack([tr[Ytr[tr]==c][:600] for c in (7,8,9)])
        dev_idx=np.concatenate([maj,mino])
        X=torch.tensor(Xtr[dev_idx],device=dev); Y=torch.nn.functional.one_hot(torch.tensor(Ytr[dev_idx],device=dev).long(),10).float()
        return X,Y,torch.tensor(Xtr[val],device=dev),torch.tensor(Ytr[val],device=dev)
    tr=tr[np.argsort(Ytr[tr],kind='stable')][:54000]
    shards=tr.reshape(40,1350)[:,:300]; sp=r.permutation(40).reshape(K,2)
    dev_idx=np.stack([np.concatenate([shards[a],shards[b]]) for a,b in sp])
    X=torch.tensor(Xtr[dev_idx],device=dev); Y=torch.nn.functional.one_hot(torch.tensor(Ytr[dev_idx],device=dev).long(),10).float()
    return X,Y,torch.tensor(Xtr[val],device=dev),torch.tensor(Ytr[val],device=dev)
def grads(W,X,Y):
    Pm=torch.softmax(torch.bmm(X,W.expand(K,-1,-1)),2)
    return torch.bmm(X.transpose(1,2),Pm-Y)/X.shape[1]   # K x 197 x 10
def acc(W,X,Y):
    pr=(X@W).argmax(1); a=(pr==Y).float().mean().item()
    rec=torch.stack([((pr==c)&(Y==c)).sum()/(Y==c).sum() for c in range(10)]).min().item()
    return a,rec
def lw0(z):
    p=torch.sqrt(torch.clamp(2*(np.e*z+1),min=0)); w=torch.where(z>-0.25,z,-1+p-p*p/3)
    for _ in range(12):
        e=torch.exp(w); f=w*e-z; w=w-f/(e*(w+1)-(w+2)*f/(2*w+2+1e-12))
    return w
def a_of_t(G,lam,beta):
    r=torch.clamp(beta*G/torch.sqrt(P*lam),max=0.999/np.sqrt(2*np.e)); x=-lw0(-2*r*r)/2
    return torch.sqrt(x*P*lam)/G
def a_of(G,lam,beta):  # smaller root of a*exp(-a^2 G^2/(P lam))=beta  (Lambert W)
    r=beta*G/np.sqrt(P*lam); r=np.minimum(r,1/np.sqrt(2*np.e))
    x=-lambertw(-2*r**2,0).real/2
    return np.sqrt(x*P*lam)/G
def run(method,lr,par,sigma,seed,Gmax=None):
    X,Y,Xv,Yv=split(seed); tg=torch.Generator(device=dev); tg.manual_seed(1000+seed)
    r=np.random.default_rng(1000+seed); lamn=np.geomspace(0.25,1.0,K); r.shuffle(lamn)
    lam=torch.tensor(lamn,device=dev,dtype=torch.float64)
    W=torch.zeros(1,197,10,device=dev); curve=[]; parts=[]
    for t in range(T):
        g=grads(W,X,Y); G=g.flatten(1).norm(dim=1).double()
        if method=='ideal': ghat=g.mean(0)
        else:
            h2=-torch.log(torch.rand(K,device=dev,generator=tg,dtype=torch.float64))*lam
            if method=='fixed': a=torch.full((K,),par,device=dev,dtype=torch.float64); thr=a**2*G**2/P; beta=torch.tensor(par,device=dev,dtype=torch.float64)
            elif method=='blind':
                beta=par*(torch.sqrt(P*lam/(2*np.e))/Gmax).min(); a=a_of_t(torch.full((K,),Gmax,device=dev,dtype=torch.float64),lam,beta); thr=a**2*Gmax**2/P
            else:
                beta=par*(torch.sqrt(P*lam/(2*np.e))/G).min(); a=a_of_t(G,lam,beta); thr=a**2*G**2/P
            sent=(h2>=thr)
            m=(sent*a).float()
            y=(m[:,None,None]*g).sum(0)+sigma*torch.randn(197,10,device=dev,generator=tg)
            ghat=y/(K*beta).float(); parts.append(sent.double().mean())
        W=W-lr*ghat.unsqueeze(0)
        if t%10==9 or t==T-1: curve.append(acc(W[0],Xv,Yv)[0])
    va=curve[-1]; ta,rec=acc(W[0],Xte_t,Yte_t)
    lg=torch.bmm(X,W.expand(K,-1,-1)); loss=-(Y*torch.log_softmax(lg,2)).sum(2).mean().item()
    minloc=(lg.argmax(2)==Y.argmax(2)).float().mean(1).min().item()
    return dict(val=va,test=ta,minrec=rec,loss=loss,minloc=minloc,part=float(torch.stack(parts).mean()) if parts else 1.0,diverged=bool(not np.isfinite(W.sum().item())))
def gmax_est():
    X,Y,_,_=split(500); W=torch.zeros(1,197,10,device=dev); m=0
    for t in range(60):
        g=grads(W,X,Y); m=max(m,g.flatten(1).norm(dim=1).max().item()); W=W-1.0*g.mean(0).unsqueeze(0)
    return 1.1*m
Gmax=gmax_est(); print('Gmax',Gmax,flush=True)
grid={'ideal':[None],'fixed':[0.05,0.1,0.2,0.4] if SCEN=='std' else [0.1,0.2,0.4,0.8],'blind':[0.5,0.99],'prop':[0.5,0.99]}
LRS=[0.5,1.0,2.0,4.0] if SCEN=='std' else [2.0,4.0]
res={'Gmax':Gmax}
for sigma in [float(sys.argv[1])]:
    res[str(sigma)]={}
    for m,ps in grid.items():
        
        best=None
        for lr in LRS:
            for p in ps:
                v=np.mean([run(m,lr,p,sigma,100+s,Gmax)['val'] for s in range(2)])
                if best is None or v>best[0]: best=(v,lr,p)
        rs=[run(m,best[1],best[2],sigma,s,Gmax) for s in range(10)]
        res[str(sigma)][m]={'lr':best[1],'par':best[2],'runs':rs}
        print(sigma,m,best,np.mean([x['test'] for x in rs]),flush=True)
    json.dump(res,open(f'/workspace/USER/comm-letter-ota/{SCEN}_{sys.argv[1]}.json','w'))
print('DONE')
