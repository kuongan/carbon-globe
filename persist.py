"""Is the model's error at a site persistent across climate periods? (INSIGHTS 4.56, prerequisite of a site-residual
correction). 97% of the test sites are grid sites with 40 years of ED history for all 15 ages. If the direct model's
out-of-fold error at a site in one period predicts its error in another period (different weather, different
starting state), the historical residual of a site can correct its future prediction.

Two periods on the same sites: P1 = y20 - y0 from (state at 0, profile of the 15 ages at 0, climate of years 0-19),
P2 = y40 - y20 from (state at 20, profile at 20, climate of years 20-39). LightGBM, 2-fold out-of-fold by 10-degree
blocks. Output: correlation of the P1 and P2 residuals of the same (site, age), and the share of P2 error variance a
P1-residual correction removes.
usage: python persist.py [n_sites]
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
C1 = np.empty((N, len(KEEP)), np.float32); C2 = np.empty((N, len(KEEP)), np.float32)
for i in range(0, N, 256):
    a = np.array(gx[idx[i:i + 256]], np.float32).mean(2)[:, :, KEEP]
    C1[i:i + len(a)] = a[:, :20].mean(1); C2[i:i + len(a)] = a[:, 20:].mean(1)
Y = np.stack([np.array(np.load(f'{D}/glob_Y{int(a):03d}.npy', mmap_mode='r')[idx][:, [0, 20, 40], 11, :], np.float32)
              for a in AG])                                                 # (15, n, 3, 7)
ok = np.isfinite(C1).all(1) & np.isfinite(C2).all(1) & np.isfinite(Y).all(axis=(0, 2, 3))
stamp(f'loaded {N} sites, ok {ok.sum()}')
blk = (r[idx] // 20) * 1000 + c[idx] // 20
fold = np.isin(blk, rng.choice(np.unique(blk), len(np.unique(blk)) // 2, replace=False)).astype(int)
P = dict(objective='l2', learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=0)


def rows(t0, clim, sites):
    st = Y[:, sites, t0]                                                   # (15, n, 7)
    prof = st[:, :, :2].transpose(1, 0, 2).reshape(len(sites), 30)
    return np.concatenate([np.concatenate([st[i], np.full((len(sites), 1), AG[i] + 20 * (t0 > 0)), prof, clim[sites]], 1)
                           for i in range(15)], 0).astype(np.float32)


res = {}
for name, t0, t1, clim in (('P1', 0, 1, C1), ('P2', 1, 2, C2)):
    e = np.full((15, N, 2), np.nan)
    for f in (0, 1):
        tr, te = np.flatnonzero(ok & (fold != f)), np.flatnonzero(ok & (fold == f))
        Xtr, Xte = rows(t0, clim, tr), rows(t0, clim, te)
        for k in (0, 1):
            ytr = (Y[:, tr, t1, k] - Y[:, tr, t0, k]).reshape(-1)
            m = lgb.train(P, lgb.Dataset(Xtr, ytr), 400)
            pred = Y[:, te, t0, k] + m.predict(Xte).reshape(15, -1)
            e[:, te, k] = (Y[:, te, t1, k] - pred) / (H if k == 0 else A_)
    res[name] = e
    stamp(f'{name} out-of-fold metric {0.5 * np.nanmean(e[..., 0] ** 2 + e[..., 1] ** 2):.4f}')

e1, e2 = res['P1'], res['P2']
alive = (Y[:, :, 0, 1] > 0) & ok[None, :]
for k, nm in ((0, 'height'), (1, 'agb')):
    a, b = e1[..., k][alive], e2[..., k][alive]
    beta = np.sum(a * b) / np.sum(a * a)
    print(f'{nm}: corr(P1 residual, P2 residual) {np.corrcoef(a, b)[0, 1]:.3f} | best scale {beta:.3f} | '
          f'P2 error removed by beta * P1 residual {1 - np.mean((b - beta * a) ** 2) / np.mean(b ** 2):.1%}')
    s1, s2 = np.nanmean(np.where(alive, e1[..., k], np.nan), 0), np.nanmean(np.where(alive, e2[..., k], np.nan), 0)
    m = np.isfinite(s1) & np.isfinite(s2)
    print(f'   site-mean residual (over ages) corr {np.corrcoef(s1[m], s2[m])[0, 1]:.3f}')
    for g, sel in (('young <=50', AG <= 50), ('old >50', AG > 50)):
        aa, bb = e1[sel][..., k][alive[sel]], e2[sel][..., k][alive[sel]]
        print(f'   {g}: corr {np.corrcoef(aa, bb)[0, 1]:.3f}')
