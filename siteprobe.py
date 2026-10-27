"""Which sites gain most from CO2 under ssp585, on top of x6 (INSIGHTS 4.42); solved exactly like dirprobe.py.
x6 already holds the age split (young up). Four site/stand patterns, each mean-zero per variable on ssp585 rows:
  warm   CO2 helps warm stands more (photorespiration): above-zero state x z(f0), f0 = temperature (corr 0.93 with
         |lat| on the grid; the test's affine map keeps z-scores)
  npp    productive stands gain more: above-zero state x z(y0 npp within the seed age)
  incr   the boost scales with the 40-year growth, not the level: (x6 year 40 - y0)
  h_only the young step on height only (does CO2 move height as much as agb?)
Each direction is orthogonalised (Gram-Schmidt, metric units) against x6's own steps (young, dry) and the ones before
it in NAMES order, so every probe measures only new information: npp = productivity beyond temperature, etc.
Probe = +0.003 on x6's LB if the direction is useless.

usage: python siteprobe.py make  |  python siteprobe.py solve <LB x6> <LB warm> <LB npp> <LB incr> <LB h_only>
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S
import dirprobe as DP
from scnprobe import demean, SC, ZERO

SCN, BASE = 'ssp585', 'x6'
NAMES = ['warm', 'npp', 'incr', 'h_only']


def z(v):
    return (v - v.mean()) / v.std()


def vectors():
    alive = ~S.DEAD[SCN]
    b = E.load(BASE, SCN)[..., :2]
    up = b - ZERO
    t = np.load(f'data/public/test_{SCN}.npz')
    y0 = t['test_y0'].astype(np.float64).transpose(1, 0, 2)               # (age, site, 7), mapped space
    D = {}
    D['warm'] = up * z(t['test_x'][:, :, :, 0].astype(np.float64).mean(axis=(1, 2)))[None, :, None]
    npp = np.zeros(alive.shape)
    for a in range(len(E.SEED_AGES)):
        npp[a, alive[a]] = z(y0[a, alive[a], 5])
    D['npp'] = up * npp[..., None]
    D['incr'] = b - y0[..., :2]
    young = (np.array(E.SEED_AGES) <= 50).astype(float)[:, None, None]
    D['h_only'] = up * young * np.array([1.0, 0.0])
    p = t['test_x'][:, :, :, 1].astype(np.float64).mean(axis=(1, 2))
    done = [demean(v / SC, alive) for v in (up * young, up * z(-p)[None, :, None])]    # x6's young, dry
    done[1] = done[1] - np.sum(done[1] * done[0]) / np.sum(done[0] ** 2) * done[0]
    D = {k: demean(v / SC, alive) for k, v in D.items()}
    for k in NAMES:
        for u in done:
            D[k] = D[k] - np.sum(D[k] * u) / np.sum(u * u) * u
        done.append(D[k])
    Q = np.array([[0.5 * np.sum(D[a] * D[b]) / E.N_ROWS for b in NAMES] for a in NAMES])
    return NAMES, D, Q


DP.vectors, DP.BASE, DP.SCN = vectors, BASE, SCN


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        names, D, Q = vectors()
        t = {k: float(np.sqrt(DP.STEP_LB / Q[i, i])) for i, k in enumerate(names)}
        for k in names:
            DP.write(t[k] * D[k], f'result/st_{k}.csv')
            print(f'wrote result/st_{k}.csv | step {t[k]:.3f}')
        json.dump(dict(base=BASE, scn=SCN, t=t, Q=Q.tolist()), open('result/dp_meta.json', 'w'), indent=1)
        print('direction correlation:\n', np.round(Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q))), 3))
    else:
        DP.solve(float(sys.argv[2]), [float(x) for x in sys.argv[3:]])
