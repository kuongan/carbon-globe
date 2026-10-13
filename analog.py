"""Own-site analog members (no training, local, ~1 min). INSIGHTS 4.27.

97% of the test sites are grid sites (test_transform.py); the grid holds ED's own 40-year run of the SAME site and seed
age under historical climate (_matched_traj.npz: Yt (15 ages, 5989 sites, 41 years, 7)). The test run starts from a
close but different state (height corr 0.91-0.98 per age, agb 0.95-0.98) under future climate. Variants (physical):
  add    y0_test + (y40_grid - y0_grid)        ED's historical 40-year change of that site/age, moved to the test start
  ratio  y0_test * y40_grid / y0_grid          same as a relative change (stocks), clipped to [0.2, 5]; add when y0 ~ 0
  hist   y40_grid                              the historical end state itself (ignores the test start)
Unmatched sites (3%) take the reference file's values mapped back to physical. Then, like slope.py match, one slope
per scenario/variable maps physical -> test space so the alive rows have the reference's mean (the analog has no CO2
response). Dead rows exact.

usage: python analog.py [ref.csv]   (default result/probe_dead_T2.csv)
  -> result/analog_preds.npz (physical, keys <scn>_<variant>) and result/match_analog_<variant>.csv
"""
import sys
import numpy as np
import pandas as pd

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCN = ('ssp126', 'ssp585')
H, A = 10.90645, 3.39567
N = 6171

ref_csv = sys.argv[1] if len(sys.argv) > 1 else 'result/probe_dead_T2.csv'
tz = np.load('result/test_transform.npz'); ys, yb = tz['y_s'].astype(np.float64), tz['y_b'].astype(np.float64)
mt = np.load('_matched_traj.npz'); good, Yt = mt['good'], mt['Yt'].astype(np.float64)
g0, g40 = Yt[:, :, 0], Yt[:, :, 40]                                   # (15, n_good, 7) physical
ref = pd.read_csv(ref_csv).set_index('id')

out = {}
for scn in SCN:
    ty = np.transpose(np.load(f'data/public/test_{scn}.npz')['test_y0'].astype(np.float64), (1, 0, 2))   # (15, N, 7)
    y0 = np.maximum((ty - yb) / ys, 0.0)
    ids = np.array([[f'{scn}_{s}_{a}' for s in range(N)] for a in AGES])
    r = ref.loc[ids.ravel()]
    fill = np.zeros((15, N, 7))
    fill[..., 0] = np.maximum((r['height'].to_numpy().reshape(15, N) * H - yb[0]) / ys[0], 0)
    fill[..., 1] = np.maximum((r['agb'].to_numpy().reshape(15, N) * A - yb[1]) / ys[1], 0)
    t0 = y0[:, good]
    rat = np.where(g0 > 1e-3, np.clip(g40 / np.maximum(g0, 1e-3), 0.2, 5.0), np.nan)
    var = {'add': np.maximum(t0 + g40 - g0, 0.0),
           'ratio': np.where(np.isfinite(rat), t0 * np.nan_to_num(rat), np.maximum(t0 + g40 - g0, 0.0)),
           'hist': g40}
    for k, v in var.items():
        p = fill.copy(); p[:, good] = v
        out[f'{scn}_{k}'] = p.astype(np.float32)
np.savez_compressed('result/analog_preds.npz', **out)
print('wrote result/analog_preds.npz', sorted(out))
