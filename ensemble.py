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
SCORED_TAGS = {'blend_global', 'blend_latest_x', 'blend_noco2_x'}
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
}
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
