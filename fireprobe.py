"""Disturbance events on the grid (INSIGHTS 4.44): can that year's climate predict a site-year agb collapse?
~22% of old-stand sites lose >30% agb in one year; the same year across seed ages (P 0.82 for seed 300 given seed
500), so it is a site-year event (ED fire / drought mortality). Label = seed-500 drop in year t -> t+1.
Features: monthly f0-f7 of year t, annual means of the other non-CO2 drivers, the previous year's monthly f0-f7,
and the seed-500 state at t. LightGBM classifier, holdout = 10-degree blocks. Local CPU, no slots.
usage: python fireprobe.py [n_sites]
"""
import sys, time
import numpy as np
import lightgbm as lgb

T0 = time.time()
D = 'data/carbonglobe/data_global'
CO2 = list(range(8, 32))
OTHER = [j for j in range(8, 136) if j not in CO2 and j != 32]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
rng = np.random.default_rng(0)
stamp = lambda m: print(f'[{(time.time() - T0) / 60:5.1f} min] {m}', flush=True)

gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
r, c = np.nonzero(np.load('data/carbonglobe/data_stats/mask2d.npy') > 0)
idx = np.sort(rng.choice(gx.shape[0], N, replace=False))
MON = np.empty((N, 40, 12, 8), np.float32); ANN = np.empty((N, 40, len(OTHER)), np.float32)
for i in range(0, N, 256):
    x = np.array(gx[idx[i:i + 256]], np.float32)
    MON[i:i + len(x)] = x[..., :8]; ANN[i:i + len(x)] = x[..., OTHER].mean(2)
Y = np.array(np.load(f'{D}/glob_Y500.npy', mmap_mode='r')[idx][:, :, 11, :], np.float32)   # (n, 41, 7)
stamp('loaded')
ok = np.isfinite(MON).all((1, 2, 3)) & np.isfinite(Y).all((1, 2))
b = Y[:, :, 1]
lab = (b[:, 1:] / np.maximum(b[:, :-1], 1e-6) < 0.7) & (b[:, :-1] > 0)        # (n, 40)
ni, ti = np.meshgrid(np.arange(N), np.arange(40), indexing='ij'); ni, ti = ni.ravel(), ti.ravel()
m = ok[ni] & (b[ni, ti] > 0)
ni, ti = ni[m], ti[m]
prev = MON[ni, np.maximum(ti - 1, 0)].reshape(len(ni), -1)
X = np.concatenate([MON[ni, ti].reshape(len(ni), -1), prev, ANN[ni, ti], Y[ni, ti], ti[:, None]], 1)
yl = lab[ni, ti].astype(np.float32)
blk = (r[idx] // 20) * 1000 + c[idx] // 20
hb = rng.choice(np.unique(blk), int(0.1 * len(np.unique(blk))), replace=False)
ho = np.isin(blk[ni], hb)
stamp(f'{len(X):,} site-years, event rate {yl.mean():.4f}, holdout {ho.mean():.2f}')
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=0)
mdl = lgb.train(P, lgb.Dataset(X[~ho], yl[~ho]), 3000, valid_sets=[lgb.Dataset(X[ho], yl[ho])],
                callbacks=[lgb.early_stopping(100, verbose=False)])
p = mdl.predict(X[ho]); y = yl[ho]
o = np.argsort(-p); tp = np.cumsum(y[o]); k = int(y.sum())
auc = (np.argsort(np.argsort(p))[y == 1].mean() - (k - 1) / 2) / (len(p) - k)
print(f'holdout AUC {auc:.3f} | precision at top-{k} {tp[k - 1] / k:.3f} | recall@p>0.5 {(p[y == 1] > 0.5).mean():.3f}'
      f' precision@p>0.5 {y[p > 0.5].mean() if (p > 0.5).any() else 0:.3f}')
for tt in (0, 5, 20):
    mm = ti[ho] >= tt
    print(f'  years >= {tt}: events {int(y[mm].sum())}, mean p on events {p[mm][y[mm] == 1].mean():.3f}, on others {p[mm][y[mm] == 0].mean():.4f}')
g = mdl.feature_importance('gain'); g = g / g.sum()
names = ([f'f{i}_m{mo}' for mo in range(12) for i in range(8)] + [f'prev_f{i}_m{mo}' for mo in range(12) for i in range(8)]
         + [f'ann{j}' for j in OTHER] + ['st_h', 'st_agb', 'st_soil', 'st_lai', 'st_gpp', 'st_npp', 'st_rh', 't'])
print('top features:', [(names[i], round(float(g[i]), 3)) for i in np.argsort(-g)[:12]])
