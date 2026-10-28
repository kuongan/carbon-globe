"""Own-site chronosequence features for the direct model (INSIGHTS 4.44). Local CPU, no slots.

Within one site the year-0 states of the 15 seed ages are ED's own growth curve for that site: the age-(a+40)
state at year 0 predicts the age-a state at year 40 (grid, no training: seed 1 0.037 vs persistence 0.926). The
test provides the same 15 initial states per site, in the same (T) space as the answer. The direct GBM so far sees
only the stand's own y0. Same split, same climate summary, with vs without:
  chrono: y0 height/agb of all 15 seed ages of the site (30) + the interpolated y0 at age a+40 (2) + its gap to y0
usage: python chrono.py [n_sites]  -> per-seed holdout metric table
"""
import sys, time
import numpy as np
import lightgbm as lgb

T0 = time.time()
D = 'data/carbonglobe/data_global'
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500], float)
CO2 = list(range(8, 32))
KEEP = [j for j in range(136) if j not in CO2 and j != 32]
H, A_ = 10.90645, 3.39567
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rng = np.random.default_rng(0)
stamp = lambda m: print(f'[{(time.time() - T0) / 60:5.1f} min] {m}', flush=True)

gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
r, c = np.nonzero(np.load('data/carbonglobe/data_stats/mask2d.npy') > 0)
idx = np.sort(rng.choice(gx.shape[0], N, replace=False))
clim = np.empty((N, len(KEEP) + 32 + 84), np.float32)
OTHER = [j for j in range(8, 136) if j not in CO2 and j != 32]
MON = np.empty((N, 40, 12, 8), np.float32); ANN = np.empty((N, 40, len(OTHER)), np.float32)
for i in range(0, N, 256):
    x = np.array(gx[idx[i:i + 256]], np.float32)                     # (b, 40, 12, 136)
    a = x.mean(2); b = len(x)
    MON[i:i + b] = x[..., :8]; ANN[i:i + b] = a[:, :, OTHER]
    clim[i:i + b] = np.concatenate([a[:, :, KEEP].mean(1), a[:, :, :8].reshape(b, 4, 10, 8).mean(2).reshape(b, 32),
                                    x[..., :7].mean(1).reshape(b, 84)], 1)
stamp('climate summary done')
YF = {a: np.array(np.load(f'{D}/glob_Y{a:03d}.npy', mmap_mode='r')[idx][:, :, 11, :], np.float32)
      for a in (100, 300, 500)}                                      # full paths for the event labels
Y = np.stack([np.array(np.load(f'{D}/glob_Y{int(a):03d}.npy', mmap_mode='r')[idx][:, [0, 40], 11, :], np.float32)
              for a in AG])                                          # (15, n, 2, 7)
ok = np.isfinite(clim).all(1) & np.isfinite(Y).all(axis=(0, 2, 3)) & np.isfinite(MON).all((1, 2, 3))
y0, y40 = Y[:, :, 0], Y[:, :, 1]
blk = (r[idx] // 20) * 1000 + c[idx] // 20                           # 10-degree blocks
ub = np.unique(blk)
hold = np.isin(blk, rng.choice(ub, int(0.1 * len(ub)), replace=False)) & ok
train = ~np.isin(blk, blk[hold]) & ok
stamp(f'sites train {train.sum()} hold {hold.sum()}')

tgt = np.minimum(AG + 40, 500)
ana = np.empty((15, N, 2), np.float32)
for i, a in enumerate(tgt):
    j = min(max(np.searchsorted(AG, a), 1), 14); w = (a - AG[j - 1]) / (AG[j] - AG[j - 1])
    ana[i] = (1 - w) * y0[j - 1, :, :2] + w * y0[j, :, :2]
allages = y0[:, :, :2].transpose(1, 0, 2).reshape(N, 30)


PE = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=0)


def ev_X(sites, t, state):
    return np.concatenate([MON[sites, t].reshape(len(sites), -1), MON[sites, np.maximum(t - 1, 0)].reshape(len(sites), -1),
                           ANN[sites, t], state, np.asarray(t, np.float32).reshape(-1, 1) * np.ones((len(sites), 1))], 1)


rows, labs = [], []
trs = np.flatnonzero(train)
for a, yf in YF.items():
    bb = yf[:, :, 1]
    lab = (bb[:, 1:] / np.maximum(bb[:, :-1], 1e-6) < 0.7) & (bb[:, :-1] > 0)
    ni, ti = [v.ravel() for v in np.meshgrid(trs, np.arange(40), indexing='ij')]
    mm = bb[ni, ti] > 0
    ni, ti = ni[mm], ti[mm]
    rows.append(ev_X(ni, ti, yf[ni, ti])); labs.append(lab[ni, ti])
clf = lgb.train(PE, lgb.Dataset(np.concatenate(rows), np.concatenate(labs).astype(np.float32)), 600)
del rows, labs
stamp('event classifier trained')
RISK = np.zeros((15, N, 5), np.float32)                              # sum, max, sum*(t/40), sum last decade, peak year/40
P40 = np.zeros((15, N, 40), np.float32)                              # every year's event probability
for i in range(15):
    for s0 in range(0, N, 4000):
        ss = np.arange(s0, min(s0 + 4000, N))
        P_ = np.stack([clf.predict(ev_X(ss, np.full(len(ss), t), y0[i, ss])) for t in range(40)], 1)   # (b, 40)
        w = np.arange(40) / 40
        P40[i, ss] = P_
        RISK[i, ss] = np.stack([P_.sum(1), P_.max(1), (P_ * w).sum(1), P_[:, 30:].sum(1), P_.argmax(1) / 40], 1)
stamp('risk features done')


def feats(sites, chrono):
    rows = []
    for i in range(15):
        f = [y0[i, sites], np.full((len(sites), 1), AG[i], np.float32), clim[sites]]
        if chrono:
            f += [allages[sites], ana[i, sites], ana[i, sites] - y0[i, sites, :2]]
        if chrono in ('fire', 'fire40'):
            f += [RISK[i, sites]]
        if chrono == 'fire40':
            f += [P40[i, sites]]
        rows.append(np.concatenate(f, 1))
    return np.concatenate(rows, 0)


tr, ho = np.flatnonzero(train), np.flatnonzero(hold)
P = dict(objective='l2', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.5,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbose=-1, seed=42)
res = {}
for chrono in ('fire', 'fire40'):
    Xtr, Xho = feats(tr, chrono), feats(ho, chrono)
    pred = np.empty((15, len(ho), 2))
    for k in (0, 1):
        ytr = (y40[:, tr, k] - y0[:, tr, k]).reshape(-1)
        yho = (y40[:, ho, k] - y0[:, ho, k]).reshape(-1)
        m = lgb.train(P, lgb.Dataset(Xtr, ytr), 4000, valid_sets=[lgb.Dataset(Xho, yho)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pred[..., k] = np.clip(y0[:, ho, k] + m.predict(Xho).reshape(15, -1), 0, None)
    dead = (y0[:, ho, 0] == 0) & (y0[:, ho, 1] == 0)
    pred[dead] = 0
    e = 0.5 * (((pred[..., 0] - y40[:, ho, 0]) / H) ** 2 + ((pred[..., 1] - y40[:, ho, 1]) / A_) ** 2)
    res[chrono] = e.mean(1)
    stamp(f'chrono={chrono}: holdout metric {e.mean():.4f}')
print('seed   fire  fire40')
for i, a in enumerate(AG):
    print(f'{int(a):4d}  {res["fire"][i]:.4f}  {res["fire40"][i]:.4f}')
print(f' all  {res["fire"].mean():.4f}  {res["fire40"].mean():.4f}')
