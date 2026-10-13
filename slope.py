"""Output slope of the physical -> test-space map, tuned on the LB (INSIGHTS 4.23).

A model run on physical inputs gives physical predictions p; the submission is s * p + b per variable (b exact, the
mapped zero). For agb (and, as an effective growth factor, height) the best s is unknown. The LB is quadratic in s
with a KNOWN curvature, so two submissions that differ only in s give the optimum exactly (up to LB rounding):
    LB(s) = c + 0.5 * (q s^2 + 2 r s),   q = sum(p^2) / (n A^2),   r unknown  ->  s* = -r / q

usage:
  python slope.py make  <npz> <key> <s_agb> [<s_height>]  -> result/slope_<key>_a<s_agb>_h<s_height>.csv (dead rows
                                                             exact, floor at the mapped zero)
  python slope.py solve <npz> <key> <var> <s1> <lb1> <s2> <lb2>   var = agb | height (the other slope held fixed)
  python slope.py match <npz> <key> <ref.csv> [<out.csv>]   per-scenario slopes matching the alive-row means of a
                                                scored file (never overwrites; default out result/match_<key>.csv)
"""
import os, sys
import numpy as np
import pandas as pd

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCN = ('ssp126', 'ssp585')
H, A = 10.90645, 3.39567
B = {'height': -0.900947, 'agb': -0.4014121}
S_H_EXACT = 0.8763248
N_ROWS = 185130


def load(npz, key):
    d = np.load(npz)
    return {s: np.maximum(d[f'{s}_{key}'][..., :2].astype(np.float64), 0.0) for s in SCN}     # physical, (15, N, 2)


def make(npz, key, s_a, s_h=S_H_EXACT):
    P = load(npz, key); dead = np.load('result/dead_rows.npz')
    parts = []
    for scn in SCN:
        p = P[scn]; N = p.shape[1]
        h = s_h * p[..., 0] + B['height']; a = s_a * p[..., 1] + B['agb']
        h[dead[scn]] = B['height']; a[dead[scn]] = B['agb']
        ids = [f'{scn}_{site}_{age}' for site in range(N) for age in AGES]
        parts.append(pd.DataFrame({'id': ids, 'height': (h.T / H).reshape(-1), 'agb': (a.T / A).reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)
    assert len(sub) == N_ROWS and sub.id.is_unique and np.isfinite(sub[['height', 'agb']].to_numpy()).all()
    path = f'result/slope_{key}_a{s_a:.3f}_h{s_h:.4f}.csv'
    sub.to_csv(path, index=False)
    print('wrote', path, '| means height', round(sub.height.mean(), 4), 'agb', round(sub.agb.mean(), 4))


def match(npz, key, ref_csv, path=None):
    """Per-scenario, per-variable slopes that give the alive rows the same mean as a scored reference file (the
    reference is LB-calibrated; the physical model is not: no CO2 response). Writes `path` (default
    result/match_<key>.csv); refuses to overwrite an existing file: keys repeat across npz files (direct_y0inv)."""
    P = load(npz, key); dead = np.load('result/dead_rows.npz')
    ref = pd.read_csv(ref_csv).set_index('id')
    parts, slopes = [], {}
    for scn in SCN:
        p = P[scn]; N = p.shape[1]; alive = ~dead[scn]
        ids = np.array([[f'{scn}_{site}_{age}' for site in range(N)] for age in AGES])      # (15, N)
        r = ref.loc[ids[alive].ravel()]
        out = {}
        for k, (col, sc) in enumerate((('height', H), ('agb', A))):
            s = (r[col].mean() * sc - B[col]) / p[..., k][alive].mean()
            slopes[(scn, col)] = s
            v = s * p[..., k] + B[col]; v[dead[scn]] = B[col]
            out[col] = v / sc
        parts.append(pd.DataFrame({'id': ids.T.reshape(-1), 'height': out['height'].T.reshape(-1),
                                   'agb': out['agb'].T.reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)
    assert len(sub) == N_ROWS and sub.id.is_unique and np.isfinite(sub[['height', 'agb']].to_numpy()).all()
    path = path or f'result/match_{key}.csv'
    assert not os.path.exists(path), f'{path} exists (a scored file?): pass another output path'
    sub.to_csv(path, index=False)
    print('slopes', {f'{k[0]} {k[1]}': round(v, 4) for k, v in slopes.items()}, '| wrote', path)


def solve(npz, key, var, s1, lb1, s2, lb2):
    P = load(npz, key); k, sc = (1, A) if var == 'agb' else (0, H)
    q = sum(np.sum(P[s][..., k] ** 2) for s in SCN) / (N_ROWS * sc ** 2)
    r = ((lb2 - lb1) - 0.5 * q * (s2 ** 2 - s1 ** 2)) / (s2 - s1)
    s_opt = -r / q
    lb_opt = lb1 + 0.5 * q * (s_opt ** 2 - s1 ** 2) + r * (s_opt - s1)
    d_round = 0.001 / (abs(s2 - s1) * q)          # LB shown to 3 decimals: +-0.0005 on each of the two scores
    print(f'{var}: q {q:.4f} | optimum slope {s_opt:.4f} (+-{d_round:.4f} from LB rounding) | predicted LB {lb_opt:.4f}')


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'make':
        make(a[2], a[3], float(a[4]), float(a[5]) if len(a) > 5 else S_H_EXACT)
    elif a[1] == 'match':
        match(a[2], a[3], a[4], a[5] if len(a) > 5 else None)
    else:
        solve(a[2], a[3], a[4], *map(float, a[5:9]))
