"""Step 1 of the CO2 plan (INSIGHTS 4.43): how much faster do stands grow in later calendar years of the grid, with
state, stand age and that year's climate held fixed? All sites share one CO2 path (358 -> 421 ppm), so the calendar
effect e(t) = CO2 effect + relaxation from the seeded initial state (time since seeding = t for every seed).
The two have different shapes: relaxation is front-loaded, CO2 accelerates late. Local CPU, no slots.

Annual step model: (state_t 7, stand age, year-t annual climate without the CO2 block, t) -> state_t+1 - state_t
for height and agb, LightGBM. e(t) = partial dependence on t (set t on every row, average the prediction).

usage: python co2est.py [n_sites]   -> result/co2est.npz + printed tables
"""
import sys, time
import numpy as np
import lightgbm as lgb

T0 = time.time()
D = 'data/carbonglobe/data_global'
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
CO2 = list(range(8, 32))
KEEP = [j for j in range(136) if j not in CO2 and j != 32]
N_SITES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
rng = np.random.default_rng(0)


def stamp(m):
    print(f'[{(time.time() - T0) / 60:5.1f} min] {m}', flush=True)


gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
idx = np.sort(rng.choice(gx.shape[0], N_SITES, replace=False))
A = np.empty((N_SITES, 40, 136), np.float32)
for i in range(0, N_SITES, 256):
    A[i:i + 256] = np.array(gx[idx[i:i + 256]], np.float32).mean(axis=2)
co2 = A[:, :, 8].mean(0)
stamp(f'climate loaded, CO2 {co2[0]:.0f} -> {co2[-1]:.0f}')
Y = np.stack([np.array(np.load(f'{D}/glob_Y{a:03d}.npy', mmap_mode='r')[idx][:, :, 11, :], np.float32)
              for a in SEED_AGES])                                   # (15, n, 41, 7) physical
ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(Y).all(axis=(0, 2, 3))
stamp(f'states loaded, sites ok {ok.sum()}/{N_SITES}')

# rows: (seed, site, year) with a living stand at t
S, N = len(SEED_AGES), N_SITES
si, ni, ti = np.meshgrid(np.arange(S), np.arange(N), np.arange(40), indexing='ij')
si, ni, ti = si.ravel(), ni.ravel(), ti.ravel()
st = Y[si, ni, ti]
m = ok[ni] & (st[:, 1] > 0)
si, ni, ti, st = si[m], ni[m], ti[m], st[m]
age = np.array(SEED_AGES, np.float32)[si] + ti
X = np.concatenate([st, age[:, None], A[ni, ti][:, KEEP], ti[:, None].astype(np.float32)], 1)
T_COL = X.shape[1] - 1
dy = Y[si, ni, ti + 1][:, :2] - st[:, :2]
hold = np.isin(ni, rng.choice(N, N // 10, replace=False))
stamp(f'{len(X):,} rows, {X.shape[1]} features, holdout {hold.mean():.2f}')

P = dict(objective='l2', learning_rate=0.08, num_leaves=127, min_data_in_leaf=500, feature_fraction=0.7,
         bagging_fraction=0.5, bagging_freq=1, max_bin=63, verbose=-1, seed=0)
out = {'co2': co2}
young = age <= 50
for k, name in enumerate(['height', 'agb']):
    res = {}
    for with_t in (True, False):
        cols = slice(None) if with_t else slice(0, T_COL)
        dtr = lgb.Dataset(X[~hold][:, cols], dy[~hold, k])
        dva = lgb.Dataset(X[hold][:, cols], dy[hold, k], reference=dtr)
        b = lgb.train(P, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        res[with_t] = (b, float(np.mean((b.predict(X[hold][:, cols]) - dy[hold, k]) ** 2)))
    b, mse_t = res[True]
    stamp(f'{name}: holdout mse with t {mse_t:.3e}, without t {res[False][1]:.3e}')
    smp = rng.choice(len(X), 60000, replace=False)
    e = np.zeros((2, 40))                       # young, old
    for t in range(40):
        Xs = X[smp].copy(); Xs[:, T_COL] = t
        p = b.predict(Xs)
        e[0, t], e[1, t] = p[young[smp]].mean(), p[~young[smp]].mean()
    base = np.array([dy[young, k].mean(), dy[~young, k].mean()])
    out[f'e_{name}'] = e
    out[f'base_{name}'] = base
    # does the late-minus-mid effect vary by site beyond the age split? (t 33-39 vs t 12-21, per row, sample)
    def at(ts):
        r = 0
        for t in ts:
            Xs = X[smp].copy(); Xs[:, T_COL] = t
            r = r + b.predict(Xs)
        return r / len(ts)
    d = at(range(33, 40)) - at(range(12, 22))
    tot = dy[smp, k]
    print(f'  {name}: late - mid effect by quintile (mean effect / mean annual change, young rows | old rows)')
    for gname, gv in (('temp f0', X[smp, 8]), ('precip f1', X[smp, 9]), ('agb', X[smp, 1]),
                      ('npp', X[smp, 5]), ('stand age', X[smp, 7])):
        cells = []
        for yo in (True, False):
            mm = young[smp] == yo
            q = np.quantile(gv[mm], [0, .2, .4, .6, .8, 1])
            g = np.clip(np.searchsorted(q, gv[mm], side='right') - 1, 0, 4)
            cells.append(' '.join(f'{100 * d[mm][g == j].mean() / max(abs(tot[mm][g == j].mean()), 1e-6):6.1f}'
                                  for j in range(5)))
        print(f'   {gname:9s} {cells[0]} | {cells[1]}')
    print(f'  {name} calendar effect, as % of mean annual change (young / old):')
    for t in range(0, 40, 3):
        print(f'   t {t:2d} CO2 {co2[t]:5.1f}  young {100 * e[0, t] / base[0]:6.1f}%  old {100 * e[1, t] / base[1]:6.1f}%')
np.savez('result/co2est.npz', **out)
stamp('wrote result/co2est.npz')
