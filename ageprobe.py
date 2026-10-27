"""Age shape of the ssp585 CO2 response on top of x6 (INSIGHTS 4.41), solved exactly like dirprobe.py.
x6 = x4f + ssp585 'young' step (ages <= 50 up, the rest down). Two finer directions, each a contrast WITHIN one half
(zero sum inside the half per variable, so it is orthogonal-ish to 'young' and to the scenario level):
  young_lo  ages 1-20 up, 30-50 down       (is the response concentrated in the youngest stands?)
  old_lo    ages 70-150 up, 200-500 down    (do mid-age stands still respond?)
Only ssp585 rows change; probe = +0.003 on x6's LB if the direction is useless.

usage: python ageprobe.py make  |  python ageprobe.py solve <LB x6> <LB young_lo> <LB old_lo>
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S
import dirprobe as DP
from scnprobe import demean, SC, ZERO

SCN, BASE = 'ssp585', 'x6'
NAMES = ['young_lo', 'old_lo']
GROUPS = {'young_lo': ((1, 10, 20), (30, 50)), 'old_lo': ((70, 100, 150), (200, 250, 300, 350, 400, 450, 500))}


def vectors():
    alive = ~S.DEAD[SCN]
    up = E.load(BASE, SCN)[..., :2] - ZERO                              # above the mapped zero
    ages = np.array(E.SEED_AGES)
    D = {}
    for k, (hi, lo) in GROUPS.items():
        mh, ml = np.isin(ages, hi), np.isin(ages, lo)
        d = np.zeros_like(up)
        for v in (0, 1):
            sh = up[mh][..., v][alive[mh]].sum()
            sl = up[ml][..., v][alive[ml]].sum()
            d[mh, :, v] = up[mh][..., v]
            d[ml, :, v] = -up[ml][..., v] * sh / sl                      # zero sum inside the half
        D[k] = demean(d / SC, alive)
    Q = np.array([[0.5 * np.sum(D[a] * D[b]) / E.N_ROWS for b in NAMES] for a in NAMES])
    return NAMES, D, Q


DP.vectors, DP.BASE, DP.SCN = vectors, BASE, SCN                       # reuse dirprobe's write / solve


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        names, D, Q = vectors()
        t = {k: float(np.sqrt(DP.STEP_LB / Q[i, i])) for i, k in enumerate(names)}
        for k in names:
            DP.write(t[k] * D[k], f'result/ap_{k}.csv')
            print(f'wrote result/ap_{k}.csv | step {t[k]:.3f}')
        json.dump(dict(base=BASE, scn=SCN, t=t, Q=Q.tolist()), open('result/dp_meta.json', 'w'), indent=1)
        print('direction correlation:\n', np.round(Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q))), 3))
    else:
        DP.solve(float(sys.argv[2]), [float(x) for x in sys.argv[3:]])
