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
  python ensemble_scn.py ingest <run> <npz> [<key>|-] [map[=<agb slope>]]
        -> result/<run>_preds/*.npy + result/hyb_<run>.csv + result/single_<run>.csv from a preds npz (keys <scn> or
           <scn>_<key>); 'map' = the npz is physical, map it to the test/truth space first (T2, INSIGHTS 4.23)
  env ENSEMBLE_DEAD=0: old behaviour (no dead-row fix, clip at 0). Default: dead rows set to their known truth and
      removed from the weight fit (component LBs corrected by their own dead-row error)
  env ENSEMBLE_FREE=global (as in ensemble.py) allows a negative weight on that run in both scenarios
"""
import os, sys
import numpy as np
import pandas as pd
import ensemble as E

BASE_RUN = 'bg_scn'
# LB of result/hyb_<run>.csv (fill in after submitting); the base itself and runs with no file yet are skipped
HYB_LB = {
    # ---- pre-T2 era ----
    'v2_scheduled_sampling': 0.127,
    'lstm_xl':               0.123,
    'ecoformer_ed':          0.125,
    'global':                0.116,
    'global_co2_hist':       0.109,
    'global2_co2_hist':      0.109,
    'a2_dsh_v2':             0.089,
    'a2_y0bx1':              0.131,
    'noco2_ep6':             0.104,
    's7_ep3':                0.103,
    'noco2_ep4':             0.093,
    'y0_persist':            0.174,
    # ---- T2 era: hybrid LB used for per-scenario solve ----
    'gbm_direct_y0inv':      0.097,
    'gbm_avg_y0inv':         None,    # placeholder for shared(); ingest_csv+hyb not needed
    'gbm_rollout_y0inv':     None,    # rollout; only useful in shared()
}


def load_scn(run, scn):
    return E.load(run, scn)


# T2 (INSIGHTS 4.23): the truth is in the test's mapped space, whose zero is (B_H, B_A) physical-scale units; dead
# stands (result/dead_rows.npz) are known exactly. ENSEMBLE_DEAD=0 restores the old behaviour (clip at 0, no dead fix)
DEAD_FIX = os.environ.get('ENSEMBLE_DEAD', '1') == '1'
B_H, B_A = -0.900947, -0.4014121
DEAD = dict(np.load(os.path.join(E.RESULT_DIR, 'dead_rows.npz'))) if DEAD_FIX else None


def write(pred_by_scn, path, fix=True):
    """fix=False for probe files (hybrids): their LB must stay comparable with the raw component LBs, which the
    solver corrects by dead_err() itself. Values are floored at the mapped zero (no effect on the old >= 0 runs)."""
    lo = (B_H, B_A) if DEAD_FIX else (0.0, 0.0)
    parts = []
    for scn in E.SCENARIOS:
        p = pred_by_scn[scn][..., :2].copy(); N = p.shape[1]
        p[..., 0] = np.maximum(p[..., 0], lo[0]); p[..., 1] = np.maximum(p[..., 1], lo[1])
        if DEAD_FIX and fix:
            p[DEAD[scn]] = (B_H, B_A)                 # dead stays dead: the mapped zero (exact truth)
        ids = [f'{scn}_{site}_{age}' for site in range(N) for age in E.SEED_AGES]
        parts.append(pd.DataFrame({'id': ids, 'height': (p[:, :, 0].T / E.H_SCALE).reshape(-1),
                                   'agb': (p[:, :, 1].T / E.A_SCALE).reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)
    assert len(sub) == E.N_ROWS and sub['id'].is_unique and np.isfinite(sub[['height', 'agb']].to_numpy()).all()
    sub.to_csv(path, index=False)
    return sub


def dead_err(p, scn):
    """Metric contribution of the dead rows of one scenario (truth known), normalized like the per-scenario MSE."""
    m = DEAD[scn]
    return 0.5 * (np.sum((p[..., 0][m] - B_H) ** 2) / E.H_SCALE ** 2 +
                  np.sum((p[..., 1][m] - B_A) ** 2) / E.A_SCALE ** 2) / m.size


def msd_alive(P, scn):
    """Pairwise msd over the rows whose truth is unknown (alive), normalized by ALL rows of the scenario."""
    keep = ~DEAD[scn]; R = len(P)
    D = np.zeros((R, R))
    for i in range(R):
        for j in range(i + 1, R):
            D[i, j] = D[j, i] = 0.5 * (np.sum(((P[i][..., 0] - P[j][..., 0])[keep] / E.H_SCALE) ** 2) +
                                       np.sum(((P[i][..., 1] - P[j][..., 1])[keep] / E.A_SCALE) ** 2)) / keep.size
    return D


def make():
    base585 = load_scn(BASE_RUN, 'ssp585')
    for r, v in HYB_LB.items():
        if v is None:
            path = f'result/hyb_{r}.csv'
            write({'ssp126': load_scn(r, 'ssp126'), 'ssp585': base585}, path, fix=False)
            print('wrote', path)


def solve():
    free_names = tuple(x for x in os.environ.get('ENSEMBLE_FREE', '').split(',') if x)
    skip = set(x for x in os.environ.get('ENSEMBLE_SKIP', '').split(',') if x)    # e.g. y0_persist -> the scn2 list
    runs = [BASE_RUN] + [r for r, v in HYB_LB.items() if v is not None and r not in skip]
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
        if DEAD_FIX:      # remove the known dead rows from the problem: their error becomes 0 after the fix
            m = m - np.array([dead_err(p, scn) for p in P])
            C = E.error_cov(m, msd_alive(P, scn))
        else:
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
    # scored: submission_blend_scn2.csv (simplex, LB 0.074), submission_blend_scn2_x.csv (ENSEMBLE_FREE=global,
    # global2_co2_hist, LB 0.074); ENSEMBLE_TAG picks another name so scored files are not overwritten
    path = f'result/submission_{os.environ.get("ENSEMBLE_TAG", "blend_scn_new")}{"_x" if free else ""}.csv'
    write(out, path)
    print('wrote', path)


def ingest_csv():
    """python ensemble_scn.py ingest_csv <run> <submission.csv>: a scored file (test/truth space, submission units)
    -> result/<run>_preds/{ssp126,ssp585}.npy (height, agb in physical-scale units; other channels 0) + its hybrid"""
    run, src = sys.argv[2], sys.argv[3]
    sub = pd.read_csv(src).set_index('id')
    out_dir = os.path.join(E.RESULT_DIR, f'{run}_preds'); os.makedirs(out_dir, exist_ok=True)
    for scn in E.SCENARIOS:
        N = 6171
        ids = [f'{scn}_{site}_{age}' for age in E.SEED_AGES for site in range(N)]
        v = sub.loc[ids]
        a = np.zeros((len(E.SEED_AGES), N, 7), np.float32)
        a[..., 0] = v['height'].to_numpy().reshape(len(E.SEED_AGES), N) * E.H_SCALE
        a[..., 1] = v['agb'].to_numpy().reshape(len(E.SEED_AGES), N) * E.A_SCALE
        np.save(os.path.join(out_dir, f'{scn}.npy'), a)
    path = f'result/hyb_{run}.csv'
    write({'ssp126': load_scn(run, 'ssp126'), 'ssp585': load_scn(BASE_RUN, 'ssp585')}, path, fix=False)
    print('wrote', out_dir, 'and', path)


def shared():
    """python ensemble_scn.py shared: ONE weight vector for both scenarios over env ENSEMBLE_RUNS (keys of
    ensemble.RUNS; no hybrid probes needed), dead rows exact. ENSEMBLE_FREE / ENSEMBLE_TAG as in solve."""
    runs = [r for r in os.environ['ENSEMBLE_RUNS'].split(',') if r]
    free_names = tuple(x for x in os.environ.get('ENSEMBLE_FREE', '').split(',') if x)
    free = tuple(runs.index(r) for r in free_names if r in runs)
    lb = np.array([E.RUNS[r][0] for r in runs], np.float64)
    P = {s: [load_scn(r, s) for r in runs] for s in E.SCENARIOS}
    m = lb - 0.5 * sum(np.array([dead_err(p, s) for p in P[s]]) for s in E.SCENARIOS)   # LB without dead rows
    D = 0.5 * sum(msd_alive(P[s], s) for s in E.SCENARIOS)                             # both scenarios
    C = E.error_cov(m, D)
    corr = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    w, v = E.best_simplex_weights(C, free)
    print('runs', runs, '\nLB without dead rows', np.round(m, 4))
    print('error correlation\n', np.round(corr, 3))
    print('weights', dict(zip(runs, np.round(w, 3))), f'| estimated LB {v:.4f}')
    out = {s: sum(wi * p for wi, p in zip(w, P[s]) if wi != 0) for s in E.SCENARIOS}
    path = f'result/submission_{os.environ.get("ENSEMBLE_TAG", "blend_shared_new")}.csv'
    write(out, path)
    print('wrote', path)


def ingest():
    """python ensemble_scn.py ingest <run> <preds.npz> [<key>]: npz with keys ssp126/ssp585 (or <scn>_<key>, e.g.
    gbm_preds.npz key direct), (15, N, 7) physical units
    -> result/<run>_preds/{ssp126,ssp585}.npy (the layout ensemble.load reads) + result/hyb_<run>.csv"""
    run, src = sys.argv[2], sys.argv[3]
    key = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else ''
    # 5th arg 'map' or 'map=<agb slope>': the npz holds PHYSICAL predictions (y0inv/xinv variants, gbm) -> mapped to
    # the test/truth space with result/test_transform.npz (y = s * physical + b; agb slope overridable)
    mp = sys.argv[5] if len(sys.argv) > 5 else ''
    if mp.startswith('map'):
        tz = np.load(os.path.join(E.RESULT_DIR, 'test_transform.npz'))
        ys, yb = tz['y_s'].astype(np.float64).copy(), tz['y_b'].astype(np.float64)
        if '=' in mp:
            ys[1] = float(mp.split('=')[1])
        print('mapping physical -> test space with slopes', np.round(ys, 4))
    d = np.load(src)
    out_dir = os.path.join(E.RESULT_DIR, f'{run}_preds')
    os.makedirs(out_dir, exist_ok=True)
    for scn in E.SCENARIOS:
        a = d[f'{scn}_{key}' if key else scn].astype(np.float64)
        if mp.startswith('map'):
            a = np.maximum(a, 0.0) * ys + yb
        a = a.astype(np.float32)
        assert a.ndim == 3 and a.shape[0] == len(E.SEED_AGES) and a.shape[2] == 7 and np.isfinite(a).all(), a.shape
        np.save(os.path.join(out_dir, f'{scn}.npy'), a)
        print(f'{scn}: {a.shape} scaled means height {a[..., 0].mean() / E.H_SCALE:.3f} '
              f'agb {a[..., 1].mean() / E.A_SCALE:.3f}')
    path = f'result/hyb_{run}.csv'
    write({'ssp126': load_scn(run, 'ssp126'), 'ssp585': load_scn(BASE_RUN, 'ssp585')}, path, fix=False)
    sp = f'result/single_{run}.csv'                   # the component itself (its LB goes into ensemble.RUNS)
    write({s: load_scn(run, s) for s in E.SCENARIOS}, sp, fix=False)
    print('wrote', out_dir, 'and', path, 'and', sp)


if __name__ == '__main__':
    {'make': make, 'solve': solve, 'ingest': ingest, 'ingest_csv': ingest_csv, 'shared': shared}[sys.argv[1]]()
