"""Rank unscored candidate members by how much a submission could improve the blend (INSIGHTS 4.31, 4.36).

For a candidate c added to the current best blend's members, the shared-weight blend estimate as a function of
LB(c) has its maximum at the PREDICTED LB (c then carries no new information). The farther the real LB lands from
it, the larger the gain. The table gives the predicted LB, the PSD-feasible range and the estimate if LB(c) misses by
+-0.01 / +-0.02: a steep curve = a slot well spent.

usage: python rank_members.py <run> [<run> ...]       (runs laid out in result/<run>_preds, e.g. by ingest)
env:   RANK_BASE (comma list, default x2's members), RANK_FREE (default lstm_xinv,gbmm_none)
"""
import os, sys
import numpy as np
import ensemble as E
import ensemble_scn as S

BASE = os.environ.get('RANK_BASE', 'dead_t2,scn2d_x,gbm_direct_y0inv,lstm_y0inv_xinv,lstm_xinv,gbmm_none,'
                                   'g1_y0inv_xinv_co2_hist').split(',')
FREE = os.environ.get('RANK_FREE', 'lstm_xinv,gbmm_none').split(',')


def rank(cand):
    runs = BASE + [cand]
    P = {s: [E.load(r, s) for r in runs] for s in E.SCENARIOS}
    D = 0.5 * sum(S.msd_alive(P[s], s) for s in E.SCENARIOS)
    de = np.array([0.5 * sum(S.dead_err(P[s][i], s) for s in E.SCENARIOS) for i in range(len(runs))])
    lb = [E.RUNS[r][0] for r in BASE]
    free = tuple(runs.index(r) for r in FREE if r in runs) + (len(BASE),)

    def est(L):
        return E.best_simplex_weights(E.error_cov(np.array(lb + [L]) - de, D), free)[1]

    Ls = np.arange(0.05, 0.35, 0.001)
    ok = np.array([np.linalg.eigvalsh(E.error_cov(np.array(lb + [L]) - de, D)).min() > -2e-4 for L in Ls])
    feas = Ls[ok]
    grid = feas[(feas > feas.min() + 0.005) & (feas < feas.max() - 0.005)]
    L0 = grid[np.argmax([est(L) for L in grid])]
    fine = np.arange(L0 - 0.001, L0 + 0.0011, 0.0001)
    L0 = fine[np.argmax([est(L) for L in fine])]
    v = {d: est(L0 + d) for d in (-0.02, -0.01, 0.01, 0.02)}
    near = min(D[-1, :-1])
    return {'cand': cand, 'pred': L0, 'lo': feas.min(), 'hi': feas.max(), 'base': est(L0), 'near': near, **v}


rows = [rank(c) for c in sys.argv[1:]]
rows.sort(key=lambda r: min(r[-0.01], r[0.01]))
print(f'base estimate (current members) {rows[0]["base"]:.4f}\n'
      f'{"candidate":28s} {"pred LB":>8s} {"feasible":>13s} {"min msd":>8s} {"-0.02":>7s} {"-0.01":>7s} {"+0.01":>7s} {"+0.02":>7s}')
for r in rows:
    print(f'{r["cand"]:28s} {r["pred"]:8.4f} {r["lo"]:6.3f}-{r["hi"]:.3f} {r["near"]:8.4f} '
          f'{r[-0.02]:7.4f} {r[-0.01]:7.4f} {r[0.01]:7.4f} {r[0.02]:7.4f}')
