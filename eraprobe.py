"""Era factor of each site as a growth direction on the final blend (INSIGHTS 4.53), solved like dirprobe.py.

97% of the test sites are grid sites. For the same site and seed age, the grid's year-0 state is the ED initial
state of 1981 and test_y0 that of 2015. Their height ratio (exact map) has median 1.00 but a strong SITE factor:
sd 0.20 in log, consistent across seed ages 10-50 (corr 0.93-0.96), agb factor corr 0.60 with it, and barely
related to the 1981->2015 climate change (|corr| <= 0.15). Hypothesis: sites whose initial stands differ from
their 1981 twins also grow differently over the next 40 years. Direction (both scenarios, alive rows of matched
sites): predicted 40-year growth of the base x z(log height factor), mean-zero per variable.

Two more directions, orthogonal to `growth` and to each other (each probe measures new information):
  level  the whole predicted state above the mapped zero x z (not only the 40-year growth)
  young  growth x z on seed ages <= 50 only (where the factor is most consistent across ages)

usage: python eraprobe.py make <base>             -> result/ep_<base>.csv (growth)
       python eraprobe.py make2 <base>            -> result/ep_<base>_level.csv, result/ep_<base>_young.csv
       python eraprobe.py solve <base> <LB base> <LB growth> [<LB level> <LB young>]
"""
import json, sys
import numpy as np
import ensemble as E
import ensemble_scn as S

SC = np.array([E.H_SCALE, E.A_SCALE])
STEP_LB = 0.003


ZERO = np.array([S.B_H, S.B_A])


def directions(base):
    """growth (as `direction`), then level and young Gram-Schmidt-orthogonalized; returns names, D, Q"""
    ef = np.load('result/era_factor.npz')
    good, lh = ef['good'], ef['log_h']
    zf = np.zeros(6171); ok = np.isfinite(lh)
    zf[good[ok]] = (lh[ok] - lh[ok].mean()) / lh[ok].std()
    young = (np.array(E.SEED_AGES) <= 50).astype(float)[:, None, None]
    raw = {'growth': {}, 'level': {}, 'young': {}}
    for s in E.SCENARIOS:
        y0t = np.load(f'data/public/test_{s}.npz')['test_y0'].astype(np.float64).transpose(1, 0, 2)[..., :2]
        pb = E.load(base, s)[..., :2]
        raw['growth'][s] = (pb - y0t) * zf[None, :, None] / SC
        raw['level'][s] = (pb - ZERO) * zf[None, :, None] / SC
        raw['young'][s] = raw['growth'][s] * young
    def demean(dd):
        for s in E.SCENARIOS:
            alive = ~S.DEAD[s]
            for k in (0, 1):
                dd[s][..., k][alive] -= dd[s][..., k][alive].mean(); dd[s][..., k][~alive] = 0.0
        return dd
    dot = lambda a, b: sum(np.sum(a[s] * b[s]) for s in E.SCENARIOS)
    names, D = ['growth', 'level', 'young'], {}
    for n in names:
        d = demean(raw[n])
        for m in D.values():
            c = dot(d, m) / dot(m, m)
            d = {s: d[s] - c * m[s] for s in E.SCENARIOS}
        D[n] = d
    Q = np.array([[0.5 * dot(D[a], D[b]) / E.N_ROWS for b in names] for a in names])
    return names, D, Q


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
    elif sys.argv[1] == 'make2':
        names, Dd, Q = directions(base)
        g = Dd['growth']
        print('growth direction identical to the first probe:',
              max(np.abs(g[s] - D[s]).max() for s in E.SCENARIOS) < 1e-9, '| corr matrix', np.round(Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q))), 3).tolist())
        meta = json.load(open(f'result/ep_{base}_meta.json'))
        for i, n in enumerate(names[1:], 1):
            t = float(np.sqrt(STEP_LB / Q[i, i]))
            S.write({s: E.load(base, s)[..., :2] + t * Dd[n][s] * SC for s in E.SCENARIOS}, f'result/ep_{base}_{n}.csv')
            meta[n] = dict(t=t, q=float(Q[i, i]))
            print(f'wrote result/ep_{base}_{n}.csv | step {t:.3f}')
        json.dump(meta, open(f'result/ep_{base}_meta.json', 'w'))
    else:
        m = json.load(open(f'result/ep_{base}_meta.json'))
        if len(sys.argv) > 5:                                         # all three (orthogonal: solved one by one)
            for n, lb in zip(('growth', 'level', 'young'), sys.argv[4:7]):
                mm = m if n == 'growth' else m[n]
                b = (float(lb) - float(sys.argv[3]) - mm['t'] ** 2 * mm['q']) / (2 * mm['t'])
                print(f'{n:6s}: b {b:.5f} | best step {-b / mm["q"]:+.3f} | gain {b * b / mm["q"]:.4f}')
            raise SystemExit
        lb0, lb = float(sys.argv[3]), float(sys.argv[4])
        b = (lb - lb0 - m['t'] ** 2 * m['q']) / (2 * m['t'])
        print(f'b {b:.5f} (+-{0.001 / (2 * m["t"]):.5f}) | best step {-b / m["q"]:+.3f} | gain {b * b / m["q"]:.4f}')
