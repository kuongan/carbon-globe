"""Blend the year-40 predictions of several trained runs into one submission.

Inputs (already unzipped): result/<run>_preds/{ssp126,ssp585}.npy (+ optional val_pred.npy), physical units, (15, N, 7).

Weights are chosen WITHOUT ground truth, using the error-covariance identity
    E|p_i - p_j|^2 = MSE_i + MSE_j - 2 * Cov(e_i, e_j)
so  C_ij = (MSE_i + MSE_j - D_ij) / 2,  and a blend w has  MSE(w) = w^T C w.
MSE_i comes from each run's leaderboard score, D_ij is measured on the test predictions.
The same identity with the VAL scores gives a val-based blend for comparison (only runs that have val_pred).
The LB is shown with 3 decimals, so the weights are also re-solved with every score jittered by +/- LB_ROUND.

Usage:  python ensemble.py
"""
import itertools, json, os
import numpy as np
import pandas as pd

RESULT_DIR = 'result'
SEED_AGES  = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCENARIOS  = ['ssp126', 'ssp585']
H_SCALE, A_SCALE = 10.90645, 3.39567          # submission scale factors (Y_MEAN[0], Y_MEAN[1])
N_ROWS = 185130
LB_ROUND = 0.0005                             # half of the last LB digit shown
# Runs allowed a NEGATIVE weight (env ENSEMBLE_FREE, comma-separated). Only use it for a run that has a scored variant
# of the same model in RUNS (e.g. 'global' next to 'global_co2_hist'): the blend then extrapolates along that pair.
# Reproduce the files in the LB ledger (INSIGHTS.md):
#   OLD6=v2_scheduled_sampling,lstm_xl,lstm_xl_co2_hist,ecoformer_ed,global,global_co2_hist
#   ENSEMBLE_RUNS=$OLD6 ENSEMBLE_TAG=blend_global                       -> submission_blend_global.csv   (LB 0.100)
#   ENSEMBLE_RUNS=$OLD6 ENSEMBLE_FREE=global ENSEMBLE_TAG=blend_global_x -> submission_blend_global_x.csv (est 0.0977)
#   ENSEMBLE_RUNS=$OLD6,global2_co2_hist,bg_scn ENSEMBLE_FREE=global ENSEMBLE_TAG=blend_latest_x -> LB 0.091
#   ENSEMBLE_RUNS=<all runs up to g2_co2x3> ENSEMBLE_TAG=blend_noco2     -> submission_blend_noco2.csv (est 0.0828)
#   same ENSEMBLE_RUNS, ENSEMBLE_FREE=global,global2_co2_hist ENSEMBLE_TAG=blend_noco2_x -> LB 0.079 (best)
#   (no env)                                                             -> submission_blend_new.csv (all RUNS)
FREE_RUNS = tuple(r for r in os.environ.get('ENSEMBLE_FREE', '').split(',') if r)
# env ENSEMBLE_RUNS (comma-separated keys of RUNS) restricts the blend; env ENSEMBLE_TAG names the output.
#   the scored 0.100 file = ENSEMBLE_RUNS of the 6 runs before global2_co2_hist / bg_scn, TAG blend_global
TAG = os.environ.get('ENSEMBLE_TAG', 'blend_new')      # -> result/submission_<TAG>.csv, ensemble_weights_<TAG>.json
# scored files must never be overwritten by a run with a different run list
SCORED_TAGS = {'blend_global', 'blend_latest_x', 'blend_noco2_x', 'blend_a2_x'}
if TAG in SCORED_TAGS and not os.environ.get('ENSEMBLE_RUNS'):
    raise SystemExit(f'TAG {TAG!r} is a scored file: set ENSEMBLE_RUNS to its original run list (see above) or pick a new tag')

# run -> (public LB score, VAL 40-year metric or None)
RUNS = {
    'v2_scheduled_sampling': (0.186, 0.03253),
    'lstm_xl':               (0.168, 0.03793),
    'lstm_xl_co2_hist':      (0.166, None),    # LSTM-XL weights, test CO2 block -> 1980-2020 history
    'ecoformer_ed':          (0.169, 0.06093),
    'global':                (0.134, None),    # LSTM-XL trained on the 51.6k-site grid (EMA epoch 4, K=1)
    'global_co2_hist':       (0.122, None),    # same weights, test CO2 -> grid-mean 1980-2020 series
    'global2_co2_hist':      (0.130, None),    # grid run 2 (EMA epoch 8, K=1 only), co2_hist
    'bg_scn':                (0.094, None),    # submission_blend_global.csv x per-scenario bias factors (a scored file)
    'noco2_ep6':             (0.109, None),    # grid LSTM-XL, CO2 block held constant, best-holdout epoch 6
    'noco2_ep3':             (0.116, None),
    'noco2_ep4':             (0.091, None),    # best single model so far
    'g2_co2x2':              (0.126, None),    # grid run 2 weights, CO2 moved 2x toward history
    'g2_co2x3':              (0.128, None),
    's7_ep3':                (0.118, None),    # run 4: no CO2, SEED 7, best-holdout epoch 3
    'noco2_ep5':             (0.133, None),
    'bx_scn':                (0.078, None),    # submission_blend_noco2_x.csv x per-scenario bias factors (a scored file)
    'a2_dsh_v2':             (0.089, None),    # noco2_ep4 weights, test v2 group shifted back to the grid mean
    'a2_y0fx1':              (0.091, None),    # noco2_ep4, initial lai/gpp/npp/rh mapped to grid level (msd 0.0002)
    'a2_y0bx1':              (0.138, None),    # noco2_ep4, initial height/agb/soil mapped to grid level (control)
    'bag1_ep6':              (0.128, None),    # bagging member m1 (70% of 10-degree blocks), best-holdout epoch 6
    'nov2_snapmean':         (0.136, None),    # run 5: no CO2 + group v2 dropped in training, mean of all snapshots
    'nov2_ep4':              (0.135, None),
    'swa_snapmean':          (0.136, None),    # run 6: fine-tune from noco2_ep4 (LR 5e-5, 8 x 0.5M), mean of snapshots
    'y0_persist':            (0.285, None),    # persistence: year-40 state = test_y0 (y0_persist.py); blend component only
    # ---- T2 era (INSIGHTS 4.23): files in the test/truth space, dead rows exact (ensemble_scn.py ingest_csv) ----
    'dead_t2':               (0.072, None),    # probe_dead_T2.csv = blend_scn2 + dead rows at the mapped zero
    'scn2d_x':               (0.072, None),    # submission_blend_scn2d_x.csv
    'gbm_direct_y0inv':      (0.108, None),    # match_direct_y0inv.csv: LightGBM direct, y0 mapped back, mean-matched
    'gbm_avg_y0inv':         (0.107, None),    # 4-seed average of direct_y0inv (estimated from msd; submit later)
    'gbm_rollout_y0inv':     (0.237, None),    # LightGBM rollout, mean-matched; NOT submitted: back-solved from t2c = 0.072 (0.226-0.248)
    'gbm_direct_inv':        (0.164, None),    # match_direct_inv.csv: same + forcing mapped back
    'lstm_y0inv':            (0.111, None),    # session E submission_global_y0inv.csv: noco2_ep4, y0 -> physical, map slopes (not mean-matched)
    't2b':                   (0.070, None),    # submission_blend_t2b.csv
    'lstm_y0inv_xinv':       (0.109, None),    # session E submission_global_y0inv_xinv.csv: + forcing mapped back
    'lstm_xinv':             (0.196, None),    # session E submission_global_xinv.csv: forcing mapped back, y0 as given
    'e4':                    (0.068, None),    # submission_blend_e4.csv: dead_t2, scn2d_x, gbm_direct, lstm_y0inv
    'x1':                    (0.066, None),    # submission_blend_x1.csv (best): + lstm_y0inv_xinv 0.735, lstm_xinv -0.361
    # session F (INSIGHTS 4.31): grid run 1 WITH CO2, y0 + forcing physical, CO2 block mapped to physical ppm
    'g1_y0inv_xinv_co2inv':  (0.119, None),    # result/global-probe/submission_global_y0inv_xinv_co2inv.csv
    'g1_y0inv_xinv_co2_hist': (0.117, None),   # same model, CO2 = grid history 1980-2020
    'g1_y0inv_xinv_co2inv0.5': (0.110, None),  # same model, half the physical CO2 increase
    'gbmm_none':             (0.169, None),    # match_gbm_none.csv: GBM direct, y0 left in test space, mean-matched
    'gbmm_y0inv2':           (0.097, None),    # match_gbm_y0inv2.csv: GBM direct, y0 pushed past physical (2 y0_p - y0_t)
    'rfm_y0inv':             (0.110, None),    # match_rf_y0inv.csv: LightGBM random-forest mode, y0 physical, mean-matched
    'rfm_none':              (0.155, None),    # match_rf_none.csv: RF twin, y0 left in test space, mean-matched
    's7_y0inv_xinv':         (0.192, None),    # session G: seed 7 ep3, y0 + forcing physical
    's7_xinv':               (0.277, None),    # session G: seed 7 ep3, forcing physical, y0 as given
    'x2':                    (0.063, None),
    'x3':                    (0.062, None),
    'x4':                    (0.062, None),    # submission_blend_x4.csv: x4f's members, only x3's two negative weights
    'x4f':                   (0.061, None),    # submission_blend_x4f.csv (best): + NN y0inv2, s42ep6/ep3 pairs
    'sw_s42ep3_xinv':        (0.254, None),    # sweep run 1 (kaggle_run1): seed 42 ep3, forcing physical
    'sw_s42ep3_y0inv_xinv':  (0.164, None),
    'sw_s42ep6_xinv':        (0.192, None),    # seed 42 ep6 (best.npz)
    'sw_s42ep6_y0inv_xinv':  (0.112, None),
    'nnm_y0inv2':            (0.113, None),
    'sw_s7ep5_xinv':         (0.247, None),    # sweep run 2: seed 7 ep5, forcing physical
    'sw_s7ep5_y0inv_xinv':   (0.140, None),
    'sw_bagep6_xinv':        (0.260, None),    # sweep run 1: bagging m1 ep6, forcing physical
    'sw_bagep6_y0inv_xinv':  (0.182, None),    # match_nn_y0inv2.csv: neural direct model, y0 past physical, mean-matched    # submission_blend_x3.csv (best): x2 + seed-7 pair    # submission_blend_x2.csv (best): LSTM pair + GBM pair + g1_co2_hist
    # own-site analogs (analog.py, INSIGHTS 4.27/4.28): mean-matched to probe_dead_T2.csv
    'analog_add':            (0.173, None),    # match_analog_add.csv: test y0 + historical 40-year change of that site/age
    'analog_hist':           (0.209, None),    # match_analog_hist.csv: historical year-40 state of that site/age
}
# scored blend files reusable as components would go here too (e.g. blend_a2_x, LB 0.075); a blend of blends adds no
# new direction, so they are left out
BLEND_ON = 'lb'   # 'lb' (tuned for the test distribution) or 'val'


def load(run, key):
    a = np.load(os.path.join(RESULT_DIR, f'{run}_preds', f'{key}.npy')).astype(np.float64)
    assert np.isfinite(a).all(), f'{run}/{key} has non-finite values'
    return a


def pair_msd(a, b):
    """Mean squared difference in competition units: 0.5*(MSE_height + MSE_agb)."""
    return 0.5 * (np.mean(((a[..., 0] - b[..., 0]) / H_SCALE) ** 2) +
                  np.mean(((a[..., 1] - b[..., 1]) / A_SCALE) ** 2))


def msd_matrix(preds):
    R = len(preds)
    D = np.zeros((R, R))
    for i, j in itertools.combinations(range(R), 2):
        D[i, j] = D[j, i] = pair_msd(preds[i], preds[j])
    return D


def error_cov(mse, D):
    mse = np.asarray(mse, np.float64)
    return 0.5 * (mse[:, None] + mse[None, :] - D)       # diagonal = mse because D_ii = 0


def best_simplex_weights(C, free=()):
    """Exact min of w^T C w over sum(w) = 1 and w >= 0 except for the indices in `free`:
    try every support set (2^R - 1 of them)."""
    R = C.shape[0]
    best_w, best_m = None, np.inf
    for k in range(1, R + 1):
        for S in itertools.combinations(range(R), k):
            S = list(S)
            v = np.linalg.lstsq(C[np.ix_(S, S)], np.ones(k), rcond=None)[0]
            if abs(v.sum()) < 1e-12:
                continue
            ws = v / v.sum()
            if any(wi < -1e-12 for i, wi in zip(S, ws) if i not in free):
                continue
            ws = np.array([wi if i in free else max(wi, 0.0) for i, wi in zip(S, ws)])
            w = np.zeros(R); w[S] = ws; w /= w.sum()
            m = float(w @ C @ w)
            if m < best_m - 1e-12:
                best_w, best_m = w, m
    return best_w, best_m


def main():
    sel = [r for r in os.environ.get('ENSEMBLE_RUNS', '').split(',') if r]
    assert all(r in RUNS for r in sel), f'unknown run in ENSEMBLE_RUNS: {sel}'
    runs = sel or list(RUNS)
    lb = np.array([RUNS[r][0] for r in runs])
    test = {r: np.concatenate([load(r, s) for s in SCENARIOS], axis=1) for r in runs}   # (15, 2N, 7)
    D_test = msd_matrix([test[r] for r in runs])

    C_lb = error_cov(lb, D_test)
    corr = C_lb / np.sqrt(np.outer(np.diag(C_lb), np.diag(C_lb)))
    print('runs:', runs)
    print('test mean squared differences:\n', np.round(D_test, 4))
    print('LB error correlation:\n', np.round(corr, 3))
    ev = np.linalg.eigvalsh(C_lb)
    print('LB covariance eigenvalues:', np.round(ev, 5))
    assert ev.min() > -1e-4, 'LB covariance clearly not PSD: check the scores'

    eq = np.ones(len(runs)) / len(runs)
    free = tuple(runs.index(r) for r in FREE_RUNS)
    w_lb, m_lb = best_simplex_weights(C_lb, free)

    # robustness to the 3-decimal LB: re-solve with every score jittered inside +/- LB_ROUND
    rng = np.random.default_rng(0)
    ws, ms, real = [], [], []
    for _ in range(500):
        lb_j = lb + rng.uniform(-LB_ROUND, LB_ROUND, len(lb))
        C_j = error_cov(lb_j, D_test)
        w_j, m_j = best_simplex_weights(C_j, free)
        ws.append(w_j); ms.append(m_j); real.append(float(w_j @ C_lb @ w_j))
    ws = np.array(ws)

    print('\nestimated LB (MSE) of candidate blends:')
    for r, s in zip(runs, lb):
        print(f'  {r:24s} {s:.4f}  (single model, observed)')
    print(f'  {"equal mean":24s} {eq @ C_lb @ eq:.4f}')
    print(f'  {"LB-tuned":24s} {m_lb:.4f}  w={dict(zip(runs, np.round(w_lb, 3)))}')
    print(f'  jittered LB (+/-{LB_ROUND}): estimate {np.percentile(ms, 5):.4f}-{np.percentile(ms, 95):.4f} (p5-p95) | '
          f'weights p5-p95: ' + ', '.join(f'{r} {np.percentile(ws[:, i], 5):.2f}-{np.percentile(ws[:, i], 95):.2f}'
                                         for i, r in enumerate(runs)))
    w_avg = ws.mean(0)
    print(f'  {"jitter-averaged weights":24s} {w_avg @ C_lb @ w_avg:.4f}  w={dict(zip(runs, np.round(w_avg, 3)))}')

    # val-based blend over the runs that have val_pred
    vruns = [r for r in runs if RUNS[r][1] is not None and
             os.path.isfile(os.path.join(RESULT_DIR, f'{r}_preds', 'val_pred.npy'))]
    w_val = None
    if len(vruns) >= 2:
        C_val = error_cov([RUNS[r][1] for r in vruns], msd_matrix([load(r, 'val_pred') for r in vruns]))
        wv, mv = best_simplex_weights(C_val)
        w_val = np.zeros(len(runs)); w_val[[runs.index(r) for r in vruns]] = wv
        print(f'  {"VAL-tuned":24s} {w_val @ C_lb @ w_val:.4f}  w={dict(zip(runs, np.round(w_val, 3)))}'
              f'  (VAL {mv:.5f}, over {vruns})')

    if BLEND_ON == 'lb':
        w = w_lb if FREE_RUNS else w_avg            # extrapolating blends: point estimate (averaging is unstable)
    else:
        w = w_val
    assert w is not None, 'no val-based blend available'
    print(f'\nusing {BLEND_ON}-tuned ({"point" if FREE_RUNS else "jitter-averaged"}) weights: '
          f'{dict(zip(runs, np.round(w, 3)))}')

    # ---- blend test predictions and write the submission ----
    parts = []
    for scn in SCENARIOS:
        p = sum(wi * load(r, scn) for wi, r in zip(w, runs) if wi != 0)
        p = np.maximum(p, 0.0)                                           # (15, N, 7) physical
        N = p.shape[1]
        # id order matches sample_submission: scenario -> site -> seed age
        ids = [f'{scn}_{site}_{age}' for site in range(N) for age in SEED_AGES]
        parts.append(pd.DataFrame({'id': ids,
                                   'height': (p[:, :, 0].T / H_SCALE).reshape(-1),
                                   'agb':    (p[:, :, 1].T / A_SCALE).reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)

    assert len(sub) == N_ROWS, len(sub)
    assert sub['id'].is_unique
    assert list(sub.columns) == ['id', 'height', 'agb']
    assert sub[['height', 'agb']].notna().all().all() and (sub[['height', 'agb']] >= 0).all().all()

    out = os.path.join(RESULT_DIR, f'submission_{TAG}.csv')
    sub.to_csv(out, index=False)
    with open(os.path.join(RESULT_DIR, f'ensemble_weights_{TAG}.json'), 'w') as f:
        json.dump({'blend_on': BLEND_ON, 'weights': dict(zip(runs, map(float, w))),
                   'estimated_lb_mse': float(w @ C_lb @ w), 'inputs': RUNS}, f, indent=2)
    print(f'\nwrote {out} ({len(sub)} rows), estimated LB {w @ C_lb @ w:.4f}')
    print('height scaled: mean %.3f std %.3f | agb scaled: mean %.3f std %.3f'
          % (sub.height.mean(), sub.height.std(), sub.agb.mean(), sub.agb.std()))


if __name__ == '__main__':
    main()
