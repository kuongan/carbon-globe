"""Era factor of each site as a growth direction on the final blend (INSIGHTS 4.53), solved like dirprobe.py.

97% of the test sites are grid sites. For the same site and seed age, the grid's year-0 state is the ED initial
state of 1981 and test_y0 that of 2015. Their height ratio (exact map) has median 1.00 but a strong SITE factor:
sd 0.20 in log, consistent across seed ages 10-50 (corr 0.93-0.96), agb factor corr 0.60 with it, and barely
related to the 1981->2015 climate change (|corr| <= 0.15). Hypothesis: sites whose initial stands differ from
their 1981 twins also grow differently over the next 40 years. Direction (both scenarios, alive rows of matched
sites): predicted 40-year growth of the base x z(log height factor), mean-zero per variable.

usage: python eraprobe.py make <base>  |  python eraprobe.py solve <base> <LB base> <LB probe>
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S

SC = np.array([E.H_SCALE, E.A_SCALE])
STEP_LB = 0.003


def direction(base):
    ef = np.load('result/era_factor.npz')
    good, lh = ef['good'], ef['log_h']
    zf = np.zeros(6171)
    ok = np.isfinite(lh)
    zf[good[ok]] = (lh[ok] - lh[ok].mean()) / lh[ok].std()
    D = {}
    for s in E.SCENARIOS:
        alive = ~S.DEAD[s]
        y0t = np.load(f'data/public/test_{s}.npz')['test_y0'].astype(np.float64).transpose(1, 0, 2)[..., :2]
        growth = E.load(base, s)[..., :2] - y0t                       # predicted 40-year change, test space
        d = growth * zf[None, :, None] / SC
        for k in (0, 1):
            d[..., k][alive] -= d[..., k][alive].mean()
            d[..., k][~alive] = 0.0
        D[s] = d
    q = sum(0.5 * np.sum(D[s] ** 2) for s in E.SCENARIOS) / E.N_ROWS
    return D, q


if __name__ == '__main__':
    base = sys.argv[2]
    D, q = direction(base)
    if sys.argv[1] == 'make':
        t = float(np.sqrt(STEP_LB / q))
        S.write({s: E.load(base, s)[..., :2] + t * D[s] * SC for s in E.SCENARIOS}, f'result/ep_{base}.csv')
        json.dump(dict(t=t, q=q), open(f'result/ep_{base}_meta.json', 'w'))
        print(f'wrote result/ep_{base}.csv | step {t:.3f} (LB +0.003 if the direction is useless)')
    else:
        m = json.load(open(f'result/ep_{base}_meta.json'))
        lb0, lb = float(sys.argv[3]), float(sys.argv[4])
        b = (lb - lb0 - m['t'] ** 2 * m['q']) / (2 * m['t'])
        print(f'b {b:.5f} (+-{0.001 / (2 * m["t"]):.5f}) | best step {-b / m["q"]:+.3f} | gain {b * b / m["q"]:.4f}')
