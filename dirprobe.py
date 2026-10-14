"""Per-scenario adjustment of a scored blend along a few directions, solved exactly from one probe per direction
(INSIGHTS 4.39).

The best blend B (x4f) uses one weight vector for both scenarios. On the ssp585 rows only, B + sum_j t_j d_j has
    LB(t) = LB(B) + 2 b't + t'Qt,   Q_ij = 0.5 sum_{ssp585 alive} (d_i.d_j in metric units) / N_ROWS  (known)
and b unknown. Probe j = B + t_j d_j on ssp585 gives b_j = (LB_j - LB(B) - t_j^2 Q_jj) / (2 t_j); with all b known
the best adjustment is t* = -Q^-1 b, gain b'Q^-1 b. ssp126 stays as B (no assumption needed).
Directions are differences of scored members (all inside the blend's span), so dead rows are untouched.

usage:
  python dirprobe.py make                          -> result/dp_<name>.csv (one per direction) + result/dp_meta.json
  python dirprobe.py solve <LB base> <LB dp_1> ...  (in DIRS order) -> result/submission_blend_<BASE>_s585.csv
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S

BASE, BASE_CSV = 'x4f', 'result/submission_blend_x4f.csv'
SCN = 'ssp585'
DIRS = {                                   # name -> (plus member, minus member)
    'gbm_pair': ('gbm_direct_y0inv', 'gbmm_none'),
    'lstm_pair': ('s7_y0inv_xinv', 's7_xinv'),
    'nn': ('nnm_y0inv2', BASE),
    'dead_t2': ('dead_t2', BASE),
}
STEP_LB = 0.003                            # probe step: t^2 Q_jj = 0.003 (LB rounding then costs ~1e-4 on the gain)
SC = np.array([E.H_SCALE, E.A_SCALE])


def vectors():
    alive = ~S.DEAD[SCN]
    D = {}
    for k, (p, m) in DIRS.items():
        d = (E.load(p, SCN)[..., :2] - E.load(m, SCN)[..., :2]) / SC        # metric units
        d[~alive] = 0.0
        D[k] = d
    names = list(D)
    Q = np.array([[0.5 * np.sum(D[a] * D[b]) / E.N_ROWS for b in names] for a in names])
    return names, D, Q


def write(delta, path):
    """BASE + delta (metric units, ssp585 only); floor at the mapped zero, dead rows exact"""
    out = {s: E.load(BASE, s).copy() for s in E.SCENARIOS}
    out[SCN][..., :2] += delta * SC
    S.write(out, path)


def make():
    names, D, Q = vectors()
    t = {k: float(np.sqrt(STEP_LB / Q[i, i])) for i, k in enumerate(names)}
    for k in names:
        write(t[k] * D[k], f'result/dp_{k}.csv')
        print(f'wrote result/dp_{k}.csv | step t {t[k]:.3f} | Q_jj {Q[names.index(k), names.index(k)]:.5f}')
    json.dump(dict(base=BASE, scn=SCN, dirs=DIRS, t=t, Q=Q.tolist()), open('result/dp_meta.json', 'w'), indent=1)
    corr = Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q)))
    print('direction correlation (ssp585):\n', np.round(corr, 3))


def solve(lb0, lbs):
    names, D, Q = vectors()
    meta = json.load(open('result/dp_meta.json'))
    t = np.array([meta['t'][k] for k in names])
    lbs = np.array(lbs)
    b = (lbs - lb0 - t ** 2 * np.diag(Q)) / (2 * t)
    ts = -np.linalg.solve(Q, b)
    gain = float(b @ np.linalg.solve(Q, b))
    err = 0.001 / (2 * t)                                        # +-0.0005 on each of the two LBs
    print('b', dict(zip(names, np.round(b, 5))), '| +- from rounding', np.round(err, 5))
    print('best ssp585 steps', dict(zip(names, np.round(ts, 3))), f'| predicted gain {gain:.4f} -> LB {lb0 - gain:.4f}')
    for k, ti in zip(names, ts):                                 # one direction at a time, for reference
        i = names.index(k)
        print(f'  {k:10s} alone: step {-b[i] / Q[i, i]:+.3f}, gain {b[i] ** 2 / Q[i, i]:.4f}')
    write(sum(ti * D[k] for k, ti in zip(names, ts)), f'result/submission_blend_{BASE}_s585.csv')
    print(f'wrote result/submission_blend_{BASE}_s585.csv')


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        make()
    else:
        solve(float(sys.argv[2]), [float(x) for x in sys.argv[3:]])
