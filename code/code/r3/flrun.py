"""Shared federated-learning harness over the OTA truncated-inversion channel.

Tasks: mnist_softmax (paper's own setup), fmnist_cnn, cifar_cnn.
One round = one local full/minibatch gradient per device, one analog aggregation.
Per-round logging of gradient norms, design participation probabilities and
realised participation (needed by experiment F).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, os, gzip, urllib.request
from otalib import gains

DATA = '/workspace/USER/r3ota/data'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
_RAW = {}   # cache the decoded datasets so per-seed Tasks are cheap


# ------------------------------------------------------------------ data
def mnist_raw():
    if 'mnist' in _RAW: return _RAW['mnist']
    os.makedirs(DATA, exist_ok=True)
    base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    def get(f):
        p = f'{DATA}/{f}'
        if not os.path.exists(p): urllib.request.urlretrieve(base + f, p)
        return gzip.open(p).read()
    xi = np.frombuffer(get('train-images-idx3-ubyte.gz'), np.uint8, offset=16).reshape(-1, 784) / 255.
    yi = np.frombuffer(get('train-labels-idx1-ubyte.gz'), np.uint8, offset=8)
    xt = np.frombuffer(get('t10k-images-idx3-ubyte.gz'), np.uint8, offset=16).reshape(-1, 784) / 255.
    yt = np.frombuffer(get('t10k-labels-idx1-ubyte.gz'), np.uint8, offset=8)
    _RAW['mnist'] = (xi, yi, xt, yt)
    return _RAW['mnist']


def tv_raw(name):
    if name in _RAW: return _RAW[name]
    import torchvision
    cls = {'cifar': torchvision.datasets.CIFAR10, 'fmnist': torchvision.datasets.FashionMNIST}[name]
    tr = cls(DATA, train=True, download=True); te = cls(DATA, train=False, download=True)
    xa = np.array(tr.data); ya = np.array(tr.targets)
    xb = np.array(te.data); yb = np.array(te.targets)
    if name == 'cifar':
        xa = xa.transpose(0, 3, 1, 2) / 255.; xb = xb.transpose(0, 3, 1, 2) / 255.
        m = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        s = np.array([0.247, 0.243, 0.261]).reshape(1, 3, 1, 1)
        xa = (xa - m) / s; xb = (xb - m) / s
    else:
        xa = xa[:, None] / 255.; xb = xb[:, None] / 255.
        xa = (xa - 0.286) / 0.353; xb = (xb - 0.286) / 0.353
    _RAW[name] = (xa.astype(np.float32), ya, xb.astype(np.float32), yb)
    return _RAW[name]


# --------------------------------------------------------------- splits
def split_classes(y, K, ncls_per_dev, n_per_dev, rng, minority=None):
    """minority = (list_of_minority_classes, n_minority_devices).
    Majority devices draw ncls_per_dev classes from the remaining classes."""
    C = int(y.max()) + 1
    byc = {c: rng.permutation(np.where(y == c)[0]) for c in range(C)}
    ptr = {c: 0 for c in range(C)}
    def take(c, n):
        i = ptr[c]; idx = byc[c][i:i + n]
        if len(idx) < n: byc[c] = rng.permutation(byc[c]); ptr[c] = 0; idx = byc[c][:n]
        ptr[c] += n; return idx
    dev, devcls = [], []
    if minority is None:
        for k in range(K):
            cs = rng.choice(C, ncls_per_dev, replace=False)
            dev.append(np.concatenate([take(c, n_per_dev // ncls_per_dev) for c in cs]))
            devcls.append(cs.tolist())
    else:
        mcls, nmin = minority
        maj = [c for c in range(C) if c not in mcls]
        for k in range(K - nmin):
            cs = rng.choice(maj, ncls_per_dev, replace=False)
            dev.append(np.concatenate([take(c, n_per_dev // ncls_per_dev) for c in cs]))
            devcls.append(cs.tolist())
        for k in range(nmin):
            # if there are fewer minority devices than minority classes, each
            # minority device holds all of them, so no class is left with no data
            cs = [mcls[k % len(mcls)]] if nmin >= len(mcls) else list(mcls)
            dev.append(np.concatenate([take(c, n_per_dev // len(cs)) for c in cs]))
            devcls.append(list(cs))
    L = min(len(d) for d in dev)
    return np.stack([d[:L] for d in dev]), devcls


# --------------------------------------------------------------- models
class CNN(nn.Module):
    def __init__(s, cin=3, nc=10, ch=32, sz=32):
        super().__init__()
        s.c1 = nn.Conv2d(cin, ch, 3, padding=1); s.n1 = nn.GroupNorm(4, ch)
        s.c2 = nn.Conv2d(ch, 2 * ch, 3, padding=1); s.n2 = nn.GroupNorm(4, 2 * ch)
        s.c3 = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1); s.n3 = nn.GroupNorm(4, 2 * ch)
        s.fc = nn.Linear(2 * ch * (sz // 8) ** 2, nc)
    def forward(s, x):
        x = Fn.max_pool2d(Fn.relu(s.n1(s.c1(x))), 2)
        x = Fn.max_pool2d(Fn.relu(s.n2(s.c2(x))), 2)
        x = Fn.max_pool2d(Fn.relu(s.n3(s.c3(x))), 2)
        return s.fc(x.flatten(1))


class Softmax(nn.Module):
    def __init__(s, d=197, nc=10):
        super().__init__(); s.fc = nn.Linear(d, nc, bias=False)
    def forward(s, x): return s.fc(x)


# --------------------------------------------------------------- task
class Task:
    def __init__(s, name, K, seed, nper=None, ncls=2, minority=None, batch=None):
        # batch=None -> full local gradient each round (the paper's MNIST setting);
        # the CNN tasks pass batch=64 explicitly.
        # the softmax task is tiny and latency-bound, so it runs on the CPU.
        s.name = name; s.K = K; s.batch = batch
        s.dev = 'cpu' if name == 'mnist_softmax' else DEV
        if s.dev == 'cpu': torch.set_num_threads(8)
        rng = np.random.default_rng(seed)
        if name == 'mnist_softmax':
            xi, yi, xt, yt = mnist_raw()
            pool = lambda x: x.reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
            f = lambda x: np.hstack([pool(x.reshape(-1, 28, 28)), np.ones((len(x), 1))]).astype(np.float32)
            Xa, Ya, Xb, Yb = f(xi), yi, f(xt), yt
            s.model = Softmax(197, 10).to(s.dev)
        else:
            key = 'cifar' if name.startswith('cifar') else 'fmnist'
            Xa, Ya, Xb, Yb = tv_raw(key)
            s.model = CNN(3 if key == 'cifar' else 1, 10, 32, 32 if key == 'cifar' else 24).to(s.dev)
            if key != 'cifar':
                Xa = np.pad(Xa, ((0, 0), (0, 0), (0, 0), (0, 0)))[:, :, 2:26, 2:26]
                Xb = Xb[:, :, 2:26, 2:26]
        nper = nper or (600 if name == 'mnist_softmax' else 500)
        idx, s.devcls = split_classes(Ya, K, ncls, nper, rng, minority)
        s.Xd = torch.tensor(Xa[idx], device=s.dev)
        s.Yd = torch.tensor(Ya[idx], device=s.dev).long()
        s.Xte = torch.tensor(Xb, device=s.dev); s.Yte = torch.tensor(Yb, device=s.dev).long()
        s.params = {k: v.detach().clone() for k, v in s.model.named_parameters()}
        s.buffers = {k: v.detach().clone() for k, v in s.model.named_buffers()}
        s.keys = list(s.params); s.shapes = [s.params[k].shape for k in s.keys]
        s.sizes = [int(np.prod(sh)) for sh in s.shapes]; s.d = int(sum(s.sizes))
        s.gen = torch.Generator(device=s.dev); s.gen.manual_seed(seed)

    def flat(s, pd): return torch.cat([pd[k].reshape(-1) for k in s.keys])
    def unflat(s, v):
        out = {}; i = 0
        for k, sh, n in zip(s.keys, s.shapes, s.sizes):
            out[k] = v[i:i + n].reshape(sh); i += n
        return out

    def grads(s, w):
        """per-device gradients, (K, d)."""
        from torch.func import functional_call, grad, vmap
        pd = s.unflat(w)
        if s.batch and s.Xd.shape[1] > s.batch:
            j = torch.randint(0, s.Xd.shape[1], (s.K, s.batch), device=s.dev, generator=s.gen)
            X = torch.gather(s.Xd.flatten(2), 1, j[:, :, None].expand(-1, -1, s.Xd.flatten(2).shape[2])
                             ).reshape(s.K, s.batch, *s.Xd.shape[2:])
            Y = torch.gather(s.Yd, 1, j)
        else:
            X, Y = s.Xd, s.Yd
        def lf(p, x, y):
            return Fn.cross_entropy(functional_call(s.model, (p, s.buffers), (x,)), y)
        gs = vmap(grad(lf), in_dims=(None, 0, 0))(pd, X, Y)
        return torch.cat([gs[k].reshape(s.K, -1) for k in s.keys], 1)

    @torch.no_grad()
    def loss_per_dev(s, w, nmax=200):
        """mean local loss of every device, vectorised over devices."""
        from torch.func import functional_call, vmap
        pd = s.unflat(w)
        X, Y = s.Xd, s.Yd
        if X.shape[1] > nmax: X, Y = X[:, :nmax], Y[:, :nmax]
        def lf(x, y):
            return Fn.cross_entropy(functional_call(s.model, (pd, s.buffers), (x,)), y)
        return vmap(lf, chunk_size=8)(X, Y).float().cpu().numpy()

    @torch.no_grad()
    def evaluate(s, w, bs=5000):
        from torch.func import functional_call
        pd = s.unflat(w); pr = []
        for i in range(0, len(s.Xte), bs):
            pr.append(functional_call(s.model, (pd, s.buffers), (s.Xte[i:i + bs],)).argmax(1))
        pr = torch.cat(pr)
        acc = (pr == s.Yte).float().mean().item()
        rec = [((pr == c) & (s.Yte == c)).sum().item() / max((s.Yte == c).sum().item(), 1)
               for c in range(10)]
        return acc, rec

    @torch.no_grad()
    def dev_acc(s, w):
        from torch.func import functional_call
        pd = s.unflat(w)
        return np.array([(functional_call(s.model, (pd, s.buffers), (s.Xd[k],)).argmax(1)
                          == s.Yd[k]).float().mean().item() for k in range(s.K)])


# --------------------------------------------------------------- run
def fl_run(task, method, lr, T, sigma, lam, P=1.0, seed=0, log_every=25,
           fix_test_norm=False, Gmax=None, qfed_q=0.0, lr_decay=0.0,
           gdesign_fn=None, **kw):
    """fix_test_norm=True -> experiment F control: the silence test uses the MEAN
    gradient norm instead of the device's own norm, everything else identical."""
    K = task.K; d = task.d
    rng = np.random.default_rng(10000 + seed)
    dv = task.dev
    tg = torch.Generator(device=dv); tg.manual_seed(20000 + seed)
    w = task.flat(task.params).clone()
    lamt = np.asarray(lam, float)
    log = dict(round=[], acc=[], rec=[], G=[], pdes=[], sent=[], beta=[], mse=[],
               gnorm=[], floss=[], energy=[])
    Ghist = []; en = 0.0
    for t in range(T):
        g = task.grads(w)
        G = g.norm(dim=1).double().cpu().numpy()
        Ghist.append(G)
        gbar = g.mean(0)
        if method == 'ideal':
            ghat = gbar; sent = np.ones(K, bool); beta = 1.0; p = np.ones(K)
        elif method == 'norm_ideal':      # normalised update, PERFECT channel
            gg = torch.tensor(G, device=dv, dtype=g.dtype).clamp_min(1e-12)
            ghat = (g / gg[:, None]).mean(0)
            sent = np.ones(K, bool); beta = 1.0; p = np.ones(K)
        elif method == 'clip_ideal':      # gradient clipping, PERFECT channel
            c = np.minimum(1.0, kw.get('clipC', 1.0) / np.maximum(G, 1e-12))
            ghat = (torch.tensor(c, device=dv, dtype=g.dtype)[:, None] * g).mean(0)
            sent = np.ones(K, bool); beta = 1.0; p = np.ones(K)
        elif method == 'qfed_ideal':
            L = task.loss_per_dev(w) ** qfed_q
            ghat = (torch.tensor(L / L.sum(), device=dv, dtype=g.dtype)[:, None] * g).sum(0) * 1.0
            sent = np.ones(K, bool); beta = 1.0; p = np.ones(K)
        else:
            kk = dict(kw)
            if method == 'cfixed':
                L = task.loss_per_dev(w) ** qfed_q; kk['cvec'] = L
            if method == 'blind': kk['Gmax'] = Gmax
            if gdesign_fn is not None: kk['Gdesign'] = gdesign_fn(t, G, Ghist)
            a, c, beta, St = gains(method, G, lamt, P, **kk)
            if fix_test_norm is False or fix_test_norm is None:
                Stest = St
            elif fix_test_norm is True:                 # control: use the mean norm
                Stest = np.full(K, float(np.mean(St)))
            else:                                       # control, scaled to match participation
                Stest = np.full(K, float(np.mean(St)) * float(fix_test_norm))
            h2 = rng.exponential(lamt)
            sent = h2 >= (a * Stest) ** 2 / P
            p = np.exp(-(a * Stest) ** 2 / (P * lamt))
            en = float(np.mean(np.where(sent, (a * c * G) ** 2 / np.maximum(h2, 1e-30), 0.0)))
            m = torch.tensor(sent * a * c, device=dv, dtype=g.dtype)
            y = (m[:, None] * g).sum(0) + sigma * torch.randn(d, device=dv, generator=tg, dtype=g.dtype)
            ghat = y / (max(sent.sum(), 1) * beta if method == 'sched' else K * beta)
        lrt = lr / (1 + lr_decay * t)
        w = w - lrt * ghat
        if not torch.isfinite(w).all():
            return dict(diverged=True, acc=0.0, rec=[0.0] * 10, log=log)
        if t % log_every == 0 or t == T - 1:
            log['round'].append(t); log['G'].append(G.tolist())
            log['pdes'].append(np.asarray(p).tolist()); log['sent'].append(sent.tolist())
            log['beta'].append(float(beta))
            log['mse'].append(float((ghat - gbar).pow(2).sum().item()))
            log['gnorm'].append(float(gbar.norm().item()))
            log['energy'].append(float(en))
            log['floss'].append(float(np.mean(task.loss_per_dev(w))))
            if t % (log_every * 4) == 0 or t == T - 1:
                a_, r_ = task.evaluate(w); log['acc'].append([t, a_]); log['rec'].append([t, r_])
    acc, rec = task.evaluate(w)
    da = task.dev_acc(w)
    return dict(diverged=False, acc=acc, rec=rec, worst_rec=float(np.min(rec)),
                rec_var=float(np.var(rec)), dev_acc_mean=float(da.mean()),
                dev_acc_min=float(da.min()), dev_acc_std=float(da.std()),
                part=float(np.mean([np.mean(s_) for s_ in log['sent']])),
                energy=float(np.mean(log['energy'])) if log['energy'] else 0.0,
                log=log, w=None)


def tune(task, method, lam, T, sigma, lrs, pars=(None,), parname=None, seeds=1, **kw):
    """pick (lr, par) by mean device TRAINING loss - no test data is used."""
    best = None
    for lr in lrs:
        for p in pars:
            k = dict(kw)
            if parname is not None and p is not None: k[parname] = p
            vals = []
            for s in range(seeds):
                r = fl_run(task, method, lr, T, sigma, lam, seed=900 + s,
                           log_every=max(T // 5, 1), **k)
                vals.append(1e9 if r['diverged'] else r['log']['floss'][-1])
            v = float(np.mean(vals))
            if best is None or v < best[0]: best = (v, lr, p)
    return best


def pick_sigma(task, lam, lr, rho=1.0, target=1.0, P=1.0, seed=0):
    """choose sigma so that the receiver-noise term d*sigma^2/(K*beta)^2 equals
    `target` times ||mean gradient||^2 at the initial point under the proposed rule."""
    from otalib import gains
    w = task.flat(task.params).clone()
    g = task.grads(w); G = g.norm(dim=1).double().cpu().numpy()
    gb = float(g.mean(0).norm().item())
    a, c, beta, St = gains('prop', G, np.asarray(lam, float), P, rho=rho)
    return float(np.sqrt(target * gb ** 2 * (task.K * beta) ** 2 / task.d)), gb, float(beta)
