"""CO2 direction from ED's own calendar effect (INSIGHTS 4.43), probed on x6 like dirprobe.py.

co2est.py: on the grid, with state, stand age and that year's climate fixed, young stands grow ~30% faster in years
33-39 (CO2 ~412) than in years 12-21 (~378). Here the same annual step model gives, for every test row, the extra
40-year growth of a 'late' vs a 'mid' calendar year along an approximate path (physical y0 -> x6's year-40 state,
physical climate of each test year). Its site/age pattern is ED's CO2 response; its size is set by the probe.
The direction is mean-zero per variable and orthogonal to x6's own young and dry steps (siteprobe.py), so the probe
measures only the pattern that x6 does not have yet.

usage: python co2dir.py make [n_sites]  |  python co2dir.py solve <LB x6> <LB co2dir>
"""
import json, sys
import numpy as np
import lightgbm as lgb
import ensemble as E
import ensemble_scn as S
import dirprobe as DP
from scnprobe import demean, SC, ZERO

SCN, BASE = 'ssp585', 'x6'
NAMES = ['co2dir']
D = 'data/carbonglobe/data_global'
CO2 = list(range(8, 32))
KEEP = [j for j in range(136) if j not in CO2 and j != 32]
LATE, MID = (33, 36, 39), (12, 16, 20)
TAUS = range(0, 40, 4)
MAPS = np.load('kaggle_upload/noco2-ep4/test_maps.npz')


def train(n_sites):
    rng = np.random.default_rng(1)
    gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
    idx = np.sort(rng.choice(gx.shape[0], n_sites, replace=False))
    A = np.empty((n_sites, 40, 136), np.float32)
    for i in range(0, n_sites, 256):
        A[i:i + 256] = np.array(gx[idx[i:i + 256]], np.float32).mean(axis=2)
    Y = np.stack([np.array(np.load(f'{D}/glob_Y{a:03d}.npy', mmap_mode='r')[idx][:, :, 11, :], np.float32)
                  for a in E.SEED_AGES])
    ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(Y).all(axis=(0, 2, 3))
    si, ni, ti = [v.ravel() for v in np.meshgrid(np.arange(15), np.arange(n_sites), np.arange(40), indexing='ij')]
    st = Y[si, ni, ti]
    m = ok[ni] & (st[:, 1] > 0)
    si, ni, ti, st = si[m], ni[m], ti[m], st[m]
    X = np.concatenate([st, (np.array(E.SEED_AGES, np.float32)[si] + ti)[:, None], A[ni, ti][:, KEEP],
                        ti[:, None].astype(np.float32)], 1)
    dy = Y[si, ni, ti + 1][:, :2] - st[:, :2]
    P = dict(objective='l2', learning_rate=0.08, num_leaves=127, min_data_in_leaf=500, feature_fraction=0.7,
             bagging_fraction=0.5, bagging_freq=1, max_bin=63, verbose=-1, seed=0)
    return [lgb.train(P, lgb.Dataset(X, dy[:, k]), 450) for k in (0, 1)]


def effect(models, scn):
    """(15, N, 2) physical: summed late-minus-mid annual growth along the approximate 40-year path"""
    t = np.load(f'data/public/test_{scn}.npz')
    y0 = np.maximum((t['test_y0'].astype(np.float64) - MAPS['y_b']) / MAPS['y_s'], 0).transpose(1, 0, 2)
    y40 = np.maximum((E.load(BASE, scn)[..., :2] - MAPS['y_b'][:2]) / MAPS['y_s'][:2], 0)
    clim = (t['test_x'].astype(np.float64).mean(axis=2) - MAPS['x_b']) / MAPS['x_s']        # (N, 40, 136) physical
    na, n = y0.shape[:2]
    out = np.zeros((na, n, 2))
    for tau in TAUS:
        w = tau / 40
        st = y0.copy()
        st[..., :2] = (1 - w) * y0[..., :2] + w * y40
        age = (np.array(E.SEED_AGES, np.float64)[:, None] + tau) * np.ones((na, n))
        base = np.concatenate([st.reshape(-1, 7), age.reshape(-1, 1),
                               np.broadcast_to(clim[None, :, tau][:, :, KEEP], (na, n, len(KEEP))).reshape(-1, len(KEEP)),
                               np.zeros((na * n, 1))], 1)
        for k, mdl in enumerate(models):
            r = 0
            for sign, ts in ((1, LATE), (-1, MID)):
                for tt in ts:
                    base[:, -1] = tt
                    r = r + sign * mdl.predict(base) / len(ts)
            out[..., k] += 4 * r.reshape(na, n)
        print(f'  {scn} tau {tau} done', flush=True)
    return out


def vectors():
    meta = json.load(open('result/co2dir_meta.json'))
    d = np.load('result/co2dir_raw.npz')[SCN]
    alive = ~S.DEAD[SCN]
    up = E.load(BASE, SCN)[..., :2] - ZERO
    young = (np.array(E.SEED_AGES) <= 50).astype(float)[:, None, None]
    p = np.load(f'data/public/test_{SCN}.npz')['test_x'][:, :, :, 1].astype(np.float64).mean(axis=(1, 2))
    z = lambda v: (v - v.mean()) / v.std()
    done = [demean(v / SC, alive) for v in (up * young, up * z(-p)[None, :, None])]
    done[1] = done[1] - np.sum(done[1] * done[0]) / np.sum(done[0] ** 2) * done[0]
    v = demean(d * MAPS['y_s'][:2] / SC, alive)
    raw = v.copy()
    for u in done:
        v = v - np.sum(v * u) / np.sum(u * u) * u
    if meta.get('print', True):
        print(f'co2dir: share kept after removing young/dry {np.sum(v * v) / np.sum(raw * raw):.2f}')
    Q = np.array([[0.5 * np.sum(v * v) / E.N_ROWS]])
    return NAMES, {'co2dir': v}, Q


DP.vectors, DP.BASE, DP.SCN = vectors, BASE, SCN

if __name__ == '__main__':
    if sys.argv[1] == 'make':
        models = train(int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
        raw = {s: effect(models, s) for s in E.SCENARIOS}
        np.savez('result/co2dir_raw.npz', **raw)
        json.dump({}, open('result/co2dir_meta.json', 'w'))
        names, Dd, Q = vectors()
        tt = float(np.sqrt(DP.STEP_LB / Q[0, 0]))
        DP.write(tt * Dd['co2dir'], 'result/co2_dir.csv')
        json.dump(dict(base=BASE, scn=SCN, t={'co2dir': tt}, Q=Q.tolist()), open('result/dp_meta.json', 'w'), indent=1)
        alive = ~S.DEAD[SCN]
        for s in E.SCENARIOS:
            r = raw[s]
            print(f'{s}: mean effect (physical) height {r[..., 0][alive].mean():.3f} m, agb {r[..., 1][alive].mean():.3f}'
                  f' | young agb {r[:5, :, 1][alive[:5]].mean():.3f}, old agb {r[5:, :, 1][alive[5:]].mean():.3f}')
        print(f'wrote result/co2_dir.csv | step {tt:.3f}')
    else:
        DP.solve(float(sys.argv[2]), [float(x) for x in sys.argv[3:]])
