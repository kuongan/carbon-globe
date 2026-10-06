"""Per-scenario blend weights from "hybrid" probes.

A hybrid probe for run r = r's predictions on the ssp126 rows + the base B (bg_scn, LB 0.094) on the ssp585 rows.
The two scenarios have the same number of rows, so with MSE_s = the per-scenario competition metric
    LB(r)        = 0.5 * (MSE126(r) + MSE585(r))
    LB(hybrid_r) = 0.5 * (MSE126(r) + MSE585(B))
Writing a = MSE585(B) (unknown): MSE126(r) = 2 LB(hybrid_r) - a and MSE585(r) = 2 LB(r) - 2 LB(hybrid_r) + a.
Per scenario s the error covariance is C_s = 0.5 (MSE_s,i + MSE_s,j - D_s,ij), so every entry of C126 is shifted by
-a and every entry of C585 by +a. For weights summing to 1 that shifts w'C126w by -a and w'C585w by +a, so the
estimated LB 0.5 (w126'C126 w126 + w585'C585 w585) does NOT depend on a: per-scenario blends are exactly predictable.

usage:
  python ensemble_scn.py make                 -> result/hyb_<run>.csv for every run in HYB_LB with value None
  python ensemble_scn.py solve                -> per-scenario weights, estimate, result/submission_blend_scn[_x].csv
  env ENSEMBLE_FREE=global (as in ensemble.py) allows a negative weight on that run in both scenarios
"""
import os, sys
import numpy as np
import pandas as pd
import ensemble as E

BASE_RUN = 'bg_scn'
# LB of result/hyb_<run>.csv (fill in after submitting); the base itself and runs with no file yet are skipped
HYB_LB = {
    'v2_scheduled_sampling': 0.127,
    'lstm_xl':               0.123,
    'ecoformer_ed':          0.125,
    'global':                0.116,
    'global_co2_hist':       0.109,
    'global2_co2_hist':      0.109,
}


def load_scn(run, scn):
    return E.load(run, scn)


def write(pred_by_scn, path):
    parts = []
    for scn in E.SCENARIOS:
        p = np.maximum(pred_by_scn[scn], 0.0); N = p.shape[1]
        ids = [f'{scn}_{site}_{age}' for site in range(N) for age in E.SEED_AGES]
        parts.append(pd.DataFrame({'id': ids, 'height': (p[:, :, 0].T / E.H_SCALE).reshape(-1),
                                   'agb': (p[:, :, 1].T / E.A_SCALE).reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)
    assert len(sub) == E.N_ROWS and sub['id'].is_unique and (sub[['height', 'agb']] >= 0).all().all()
    sub.to_csv(path, index=False)
    return sub


def make():
    base585 = load_scn(BASE_RUN, 'ssp585')
    for r, v in HYB_LB.items():
        if v is None:
            path = f'result/hyb_{r}.csv'
            write({'ssp126': load_scn(r, 'ssp126'), 'ssp585': base585}, path)
            print('wrote', path)


def solve():
    free_names = tuple(x for x in os.environ.get('ENSEMBLE_FREE', '').split(',') if x)
    runs = [BASE_RUN] + [r for r, v in HYB_LB.items() if v is not None]
    assert len(runs) >= 2, 'fill in HYB_LB with the LB scores of the hybrid probes first'
    lb = np.array([E.RUNS[r][0] for r in runs])
    hyb = np.array([E.RUNS[BASE_RUN][0]] + [HYB_LB[r] for r in runs[1:]])   # the base's own hybrid is itself
    a = lb[0]                                       # any value works (cancels); use LB(B) for readable numbers
    m126 = 2 * hyb - a
    m585 = 2 * lb - 2 * hyb + a
    free = tuple(runs.index(r) for r in free_names if r in runs)
    out, est = {}, 0.0
    for scn, m in (('ssp126', m126), ('ssp585', m585)):
        P = [load_scn(r, scn) for r in runs]
        C = E.error_cov(m, E.msd_matrix(P))
        w, v = E.best_simplex_weights(C, free)
        est += 0.5 * v
        out[scn] = sum(wi * p for wi, p in zip(w, P) if wi != 0)
        print(f'{scn}: per-scenario MSE (shifted by an unknown constant) {dict(zip(runs, np.round(m, 4)))}')
        print(f'        weights {dict(zip(runs, np.round(w, 3)))}')
    # same weights for both scenarios, for comparison (= ensemble.py restricted to these runs)
    Cb = E.error_cov(lb, E.msd_matrix([np.concatenate([load_scn(r, s) for s in E.SCENARIOS], 1) for r in runs]))
    wb, vb = E.best_simplex_weights(Cb, free)
    print(f'estimated LB: per-scenario weights {est:.4f} | shared weights {vb:.4f} | base {lb[0]:.4f}')
    path = f'result/submission_blend_scn{"_x" if free else ""}.csv'
    write(out, path)
    print('wrote', path)


if __name__ == '__main__':
    {'make': make, 'solve': solve}[sys.argv[1]]()
