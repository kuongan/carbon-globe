"""Feature lab for the GBM direct model (PLAN master plan, track A): same 20k-site subset, same 10-degree block
holdout as chrono.py, feature blocks switched on cumulatively. Local CPU, no slots.

blocks (per (site, seed age) row):
  base    own y0 (7), seed age, climate summary (40-year means of 111 drivers, decade means f0-f7, monthly f0-f6)
  chrono  y0 height/agb of all 15 ages, y0 interpolated at age a+40, its gap to y0           (INSIGHTS 4.44)
  fire    event risk at the y0 state: sum / max / time-weighted / last decade / peak year     (INSIGHTS 4.44)
  fire40  every year's event probability                                                     (A1)
  prof7   all 7 state variables of the 15-age profile, interpolated at log(a+20) / log(a+40), log-age slope (A2)
  ratio   CUE = npp/gpp, NEP = npp - rh, agb/height, lai/height, y0/(age+5)                  (A3)
  climfp  per driver (annual means): 40-year slope, year-to-year std, first-5 / last-5 means  (A4)
  firepath event risk along an approximate path: state at year t = the site's own y0 profile at age a+t (A5)
  mag     expected one-year agb loss (regression, not only P(drop > 30%)) at the y0 state: sum, max, last
          decade, expected surviving fraction prod(1 - loss)                                    (A6)
  magpath the same along the path                                                              (A5 + A6)

usage: python featlab.py <n_sites> <config> [<config> ...]   config = blocks joined by '+', e.g. base+chrono+fire
"""
import sys, time
import numpy as np
import lightgbm as lgb

T0 = time.time()
D = 'data/carbonglobe/data_global'
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500], float)
LAG = np.log(AG)
CO2 = list(range(8, 32))
KEEP = [j for j in range(136) if j not in CO2 and j != 32]
OTHER = [j for j in range(8, 136) if j not in CO2 and j != 32]
H, A_ = 10.90645, 3.39567
N = int(sys.argv[1])
CONFIGS = sys.argv[2:]
NEED = set('+'.join(CONFIGS).split('+'))
rng = np.random.default_rng(0)
stamp = lambda m: print(f'[{(time.time() - T0) / 60:5.1f} min] {m}', flush=True)

# ---- load (identical draws to chrono.py: same sites, same holdout) ----
gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
r, c = np.nonzero(np.load('data/carbonglobe/data_stats/mask2d.npy') > 0)
idx = np.sort(rng.choice(gx.shape[0], N, replace=False))
clim = np.empty((N, len(KEEP) + 32 + 84), np.float32)
MON = np.empty((N, 40, 12, 8), np.float32); ANN = np.empty((N, 40, len(OTHER)), np.float32)
ANNK = np.empty((N, 40, len(KEEP)), np.float32)
for i in range(0, N, 256):
    x = np.array(gx[idx[i:i + 256]], np.float32)
    a = x.mean(2); b = len(x)
    MON[i:i + b] = x[..., :8]; ANN[i:i + b] = a[:, :, OTHER]; ANNK[i:i + b] = a[:, :, KEEP]
    clim[i:i + b] = np.concatenate([a[:, :, KEEP].mean(1), a[:, :, :8].reshape(b, 4, 10, 8).mean(2).reshape(b, 32),
                                    x[..., :7].mean(1).reshape(b, 84)], 1)
stamp('climate loaded')
YF = {a: np.array(np.load(f'{D}/glob_Y{a:03d}.npy', mmap_mode='r')[idx][:, :, 11, :], np.float32) for a in (100, 300, 500)}
Y = np.stack([np.array(np.load(f'{D}/glob_Y{int(a):03d}.npy', mmap_mode='r')[idx][:, [0, 40], 11, :], np.float32)
              for a in AG])                                          # (15, n, 2, 7)
ok = (np.isfinite(clim).all(1) & np.isfinite(Y).all(axis=(0, 2, 3)) & np.isfinite(MON).all((1, 2, 3))
      & np.isfinite(ANNK).all((1, 2)))
y0, y40 = Y[:, :, 0], Y[:, :, 1]
blk = (r[idx] // 20) * 1000 + c[idx] // 20
ub = np.unique(blk)
hold = np.isin(blk, rng.choice(ub, int(0.1 * len(ub)), replace=False)) & ok
train = ~np.isin(blk, blk[hold]) & ok
stamp(f'sites train {train.sum()} hold {hold.sum()}')


def interp_log(prof, target_lag):
    """prof (15, n, V) along LAG -> values at target_lag (15,) per row, (15, n, V); clamped at the ends"""
    out = np.empty_like(prof)
    for i, t in enumerate(target_lag):
        j = min(max(int(np.searchsorted(LAG, t)), 1), 14); w = np.clip((t - LAG[j - 1]) / (LAG[j] - LAG[j - 1]), 0, 1)
        out[i] = (1 - w) * prof[j - 1] + w * prof[j]
    return out


BLK = {}
tgt = np.minimum(AG + 40, 500)
ana = np.empty((15, N, 2), np.float32)
for i, a in enumerate(tgt):
    j = min(max(np.searchsorted(AG, a), 1), 14); w = (a - AG[j - 1]) / (AG[j] - AG[j - 1])
    ana[i] = (1 - w) * y0[j - 1, :, :2] + w * y0[j, :, :2]
if 'chrono' in NEED:
    allages = np.broadcast_to(y0[:, :, :2].transpose(1, 0, 2).reshape(1, N, 30), (15, N, 30))
    BLK['chrono'] = np.concatenate([allages, ana, ana - y0[:, :, :2]], 2)
if 'prof7' in NEED:
    p7 = np.broadcast_to(y0[:, :, 2:].transpose(1, 0, 2).reshape(1, N, 75), (15, N, 75))   # the 5 other variables
    i20 = interp_log(y0, np.log(AG + 20)); i40 = interp_log(y0, np.log(np.minimum(AG + 40, 600)))
    jp = np.maximum(np.arange(15) - 1, 0); jn = np.minimum(np.arange(15) + 1, 14)
    slope = (y0[jn] - y0[jp]) / (LAG[jn] - LAG[jp])[:, None, None]
    BLK['prof7'] = np.concatenate([p7, i20, i40 - y0, slope], 2)
if 'ratio' in NEED:
    g, n_, rh, h, ab, lai = y0[..., 4], y0[..., 5], y0[..., 6], y0[..., 0], y0[..., 1], y0[..., 3]
    BLK['ratio'] = np.stack([n_ / (g + 0.05), n_ - rh, ab / (h + 0.5), lai / (h + 0.5)], 2)
    BLK['ratio'] = np.concatenate([BLK['ratio'], y0[..., :2] / (AG[:, None, None] + 5)], 2)
if 'climfp' in NEED:
    tt = np.arange(40) - 19.5
    slope_c = (ANNK * tt[None, :, None]).sum(1) / (tt ** 2).sum()
    fp = np.concatenate([slope_c, ANNK.std(1), ANNK[:, :5].mean(1), ANNK[:, -5:].mean(1)], 1)
    BLK['climfp'] = np.broadcast_to(fp[None], (15, N, fp.shape[1]))
if NEED & {'fire', 'fire40', 'firepath', 'mag', 'magpath'}:
    PE = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=0)

    def ev_X(sites, t, state):
        return np.concatenate([MON[sites, t].reshape(len(sites), -1), MON[sites, np.maximum(t - 1, 0)].reshape(len(sites), -1),
                               ANN[sites, t], state, np.asarray(t, np.float32).reshape(-1, 1) * np.ones((len(sites), 1))], 1)
    rows, labs, loss = [], [], []
    trs = np.flatnonzero(train)
    for a, yf in YF.items():
        bb = yf[:, :, 1]
        lab = (bb[:, 1:] / np.maximum(bb[:, :-1], 1e-6) < 0.7) & (bb[:, :-1] > 0)
        lo = np.clip(1 - bb[:, 1:] / np.maximum(bb[:, :-1], 1e-6), 0, 1)        # one-year agb loss fraction
        ni, ti = [v.ravel() for v in np.meshgrid(trs, np.arange(40), indexing='ij')]
        mm = bb[ni, ti] > 0
        ni, ti = ni[mm], ti[mm]
        rows.append(ev_X(ni, ti, yf[ni, ti])); labs.append(lab[ni, ti]); loss.append(lo[ni, ti])
    XE = np.concatenate(rows); del rows
    clf = lgb.train(PE, lgb.Dataset(XE, np.concatenate(labs).astype(np.float32)), 600)
    reg = None
    if NEED & {'mag', 'magpath'}:
        reg = lgb.train(dict(PE, objective='l2'), lgb.Dataset(XE, np.concatenate(loss).astype(np.float32)), 600)
    del XE, labs, loss
    RISK = np.zeros((15, N, 5), np.float32); P40 = np.zeros((15, N, 40), np.float32)
    w = np.arange(40) / 40
    for i in range(15):
        for s0 in range(0, N, 4000):
            ss = np.arange(s0, min(s0 + 4000, N))
            P_ = np.stack([clf.predict(ev_X(ss, np.full(len(ss), t), y0[i, ss])) for t in range(40)], 1)
            P40[i, ss] = P_
            RISK[i, ss] = np.stack([P_.sum(1), P_.max(1), (P_ * w).sum(1), P_[:, 30:].sum(1), P_.argmax(1) / 40], 1)
    BLK['fire'], BLK['fire40'] = RISK, P40
    stamp('event risk done')

    def path_state(i, ss, t):
        """state of seed age i at year t ~ the site's own y0 profile at age AG[i] + t (log-age interpolation)"""
        tl = np.log(min(AG[i] + t, 500.0))
        j = min(max(int(np.searchsorted(LAG, tl)), 1), 14); w_ = np.clip((tl - LAG[j - 1]) / (LAG[j] - LAG[j - 1]), 0, 1)
        return (1 - w_) * y0[j - 1, ss] + w_ * y0[j, ss]

    def mag_summ(L):
        return np.stack([L.sum(1), L.max(1), L[:, 30:].sum(1), np.prod(1 - np.clip(L, 0, 1), 1)], 1)
    if 'firepath' in NEED:
        FP = np.zeros((15, N, 5), np.float32)
        for i in range(15):
            for s0 in range(0, N, 4000):
                ss = np.arange(s0, min(s0 + 4000, N))
                P_ = np.stack([clf.predict(ev_X(ss, np.full(len(ss), t), path_state(i, ss, t))) for t in range(40)], 1)
                FP[i, ss] = np.stack([P_.sum(1), P_.max(1), (P_ * w).sum(1), P_[:, 30:].sum(1), P_.argmax(1) / 40], 1)
        BLK['firepath'] = FP
        stamp('event risk along the path done')
    for name, along in (('mag', False), ('magpath', True)):
        if name in NEED:
            MG = np.zeros((15, N, 4), np.float32)
            for i in range(15):
                for s0 in range(0, N, 4000):
                    ss = np.arange(s0, min(s0 + 4000, N))
                    L = np.stack([reg.predict(ev_X(ss, np.full(len(ss), t), path_state(i, ss, t) if along else y0[i, ss]))
                                  for t in range(40)], 1)
                    MG[i, ss] = mag_summ(L)
            BLK[name] = MG
            stamp(f'{name} done')


def feats(sites, blocks):
    rows = []
    for i in range(15):
        f = [y0[i, sites], np.full((len(sites), 1), AG[i], np.float32), clim[sites]]
        f += [BLK[b][i, sites] for b in blocks if b not in ('base', 'anchor')]
        rows.append(np.concatenate(f, 1).astype(np.float32))
    return np.concatenate(rows, 0)


tr, ho = np.flatnonzero(train), np.flatnonzero(hold)
P = dict(objective='l2', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.5,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbose=-1, seed=42)
res = {}
for cfg in CONFIGS:
    blocks = cfg.split('+')
    Xtr, Xho = feats(tr, blocks), feats(ho, blocks)
    pred = np.empty((15, len(ho), 2))
    anchor = 'anchor' in blocks                        # target = y40 - y0 at age a+40 (own chronosequence), not - own y0
    for k in (0, 1):
        base = ana[..., k] if anchor else y0[..., k]
        ytr = (y40[:, tr, k] - base[:, tr]).reshape(-1)
        yho = (y40[:, ho, k] - base[:, ho]).reshape(-1)
        m = lgb.train(P, lgb.Dataset(Xtr, ytr), 4000, valid_sets=[lgb.Dataset(Xho, yho)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pred[..., k] = np.clip(base[:, ho] + m.predict(Xho).reshape(15, -1), 0, None)
    dead = (y0[:, ho, 0] == 0) & (y0[:, ho, 1] == 0)
    pred[dead] = 0
    e = 0.5 * (((pred[..., 0] - y40[:, ho, 0]) / H) ** 2 + ((pred[..., 1] - y40[:, ho, 1]) / A_) ** 2)
    res[cfg] = e.mean(1)
    stamp(f'{cfg}: {Xtr.shape[1]} features | holdout {e.mean():.4f}')
    del Xtr, Xho
print('seed  ' + '  '.join(f'{k[-14:]:>14s}' for k in res))
for i, a in enumerate(AG):
    print(f'{int(a):4d}  ' + '  '.join(f'{v[i]:14.4f}' for v in res.values()))
print(' all  ' + '  '.join(f'{v.mean():14.4f}' for v in res.values()))
