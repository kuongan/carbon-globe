"""Age x scenario re-check on a new best blend (PLAN track D2): the young-stand step (ages <= 50 up, the rest down,
mean-zero per variable) that gave x6 its 0.0011, re-probed on <base> one scenario at a time, solved like dirprobe.py.
A better base model may need a different correction (the chrono members already know the young growth curve).

usage: python youngprobe.py make <base>  |  python youngprobe.py solve <base> <scn> <LB base> <LB probe>
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S
import dirprobe as DP
from scnprobe import demean, SC, ZERO


def vec(base, scn):
    alive = ~S.DEAD[scn]
    up = E.load(base, scn)[..., :2] - ZERO
    young = (np.array(E.SEED_AGES) <= 50).astype(float)[:, None, None]
    d = demean(up * young / SC, alive)
    return d, 0.5 * np.sum(d * d) / E.N_ROWS


if __name__ == '__main__':
    base = sys.argv[2]
    DP.BASE = base
    if sys.argv[1] == 'make':
        meta = {}
        for scn in E.SCENARIOS:
            d, q = vec(base, scn)
            t = float(np.sqrt(DP.STEP_LB / q))
            DP.SCN = scn
            DP.write(t * d, f'result/yp_{base}_{scn}.csv')
            meta[scn] = dict(t=t, q=q)
            print(f'wrote result/yp_{base}_{scn}.csv | step {t:.3f}')
        json.dump(meta, open(f'result/yp_{base}_meta.json', 'w'), indent=1)
    else:
        scn, lb0, lb = sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
        m = json.load(open(f'result/yp_{base}_meta.json'))[scn]
        b = (lb - lb0 - m['t'] ** 2 * m['q']) / (2 * m['t'])
        err = 0.001 / (2 * m['t'])
        print(f'{scn}: b {b:.5f} (+-{err:.5f}) | best step {-b / m["q"]:+.3f} | gain {b * b / m["q"]:.4f}')
