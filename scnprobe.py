"""Site pattern of the ssp585 - ssp126 difference: four physical hypotheses, solved exactly like dirprobe.py
(INSIGHTS 4.40). Only ssp585 rows change; every direction is mean-zero per variable (the scenario level is already
LB-calibrated), so a probe measures the PATTERN only.
  uniform  ssp585 = ssp126 x (alive mean ratio per age and variable): drop the models' site-level scenario pattern
  co2lstm  the CO2-aware LSTM's (session F) CO2-only effect: (co2inv - co2_hist) in ssp585 minus the same in ssp126
  dry      CO2 helps water-limited stands more: ssp585 above-zero state x (dryness z-score of f1 = precipitation)
  young    young stands (age <= 50) respond more: their ssp585 above-zero state up, the rest down

usage: python scnprobe.py make  |  python scnprobe.py solve <LB base> <LB uniform> <LB co2lstm> <LB dry> <LB young>
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S
import dirprobe as DP

SCN, BASE = 'ssp585', 'x4f'
NAMES = ['uniform', 'co2lstm', 'dry', 'young']
SC = np.array([E.H_SCALE, E.A_SCALE])
ZERO = np.array([S.B_H, S.B_A])


def demean(d, alive):
    d = d.copy()
    for k in (0, 1):
        d[..., k][alive] -= d[..., k][alive].mean()
        d[..., k][~alive] = 0.0
    return d


def vectors():
    alive = ~S.DEAD[SCN]
    b5, b1 = E.load(BASE, 'ssp585')[..., :2], E.load(BASE, 'ssp126')[..., :2]
    up = b5 - ZERO                                                     # above the mapped zero
    D = {}
    ratio = np.ones((len(E.SEED_AGES), 1, 2))
    for a in range(len(E.SEED_AGES)):
        m = alive[a]
        ratio[a, 0] = (b5[a][m] - ZERO).mean(0) / (b1[a][m] - ZERO).mean(0)
    D['uniform'] = ZERO + (b1 - ZERO) * ratio - b5
    g = lambda r, s: E.load(r, s)[..., :2]
    D['co2lstm'] = ((g('g1_y0inv_xinv_co2inv', 'ssp585') - g('g1_y0inv_xinv_co2_hist', 'ssp585'))
                    - (g('g1_y0inv_xinv_co2inv', 'ssp126') - g('g1_y0inv_xinv_co2_hist', 'ssp126')))
    p = np.load('data/public/test_ssp585.npz')['test_x'][:, :, :, 1].astype(np.float64).mean(axis=(1, 2))
    dry = -(p - p.mean()) / p.std()
    D['dry'] = up * dry[None, :, None]
    young = (np.array(E.SEED_AGES) <= 50).astype(float)[:, None, None]
    D['young'] = up * young
    D = {k: demean(v / SC, alive) for k, v in D.items()}               # metric units, mean-zero pattern
    Q = np.array([[0.5 * np.sum(D[a] * D[b]) / E.N_ROWS for b in NAMES] for a in NAMES])
    return NAMES, D, Q


DP.vectors = vectors                                                   # reuse dirprobe's make / solve


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        names, D, Q = vectors()
        t = {k: float(np.sqrt(DP.STEP_LB / Q[i, i])) for i, k in enumerate(names)}
        for k in names:
            DP.write(t[k] * D[k], f'result/sp_{k}.csv')
            print(f'wrote result/sp_{k}.csv | step {t[k]:.3f}')
        json.dump(dict(base=BASE, scn=SCN, t=t, Q=Q.tolist()), open('result/dp_meta.json', 'w'), indent=1)
        print('direction correlation:\n', np.round(Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q))), 3))
    else:
        DP.solve(float(sys.argv[2]), [float(x) for x in sys.argv[3:]])
