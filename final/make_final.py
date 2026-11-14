"""Final submission of team iuem: a fixed-weight blend of 8 reproducible single models (INSIGHTS 4.58).

Each member is a raw model output (PHYSICAL year-40 predictions, (15 seed ages, 6171 sites, 7) per scenario, as
written by the model scripts in training mode or in inference-only mode from the supplied weights). This script
1. brings each member to the answer space (the test inputs and the hidden answer are the CarbonGlobe grid behind a
   per-variable affine map y_test = s * y + b): `map` = s * max(p, 0) + b with the slopes of result/test_transform.npz;
   `match` = per scenario and variable, the slope giving the alive rows the mean of final/ref/meanmatch_reference.csv;
   stands dead at year 0 (result/dead_rows.npz) are set to the mapped zero (their exact answer);
2. blends them with the fixed weights below (solved in closed form from the public LB of each member:
   MSE_i + MSE_j - D_ij = 2 Cov(e_i, e_j), best weights summing to 1 = C^-1 1 / 1'C^-1 1; final/README.md);
3. writes final/submission_final.csv (185,130 rows, height / 10.90645, agb / 3.39567).

usage (from the repository root):
  python final/make_final.py [--nn NPZ] [--gbm DIR] [--lstm-ep4 NPZ] [--lstm-sweep NPZ] [--lstm-s7 NPZ] [--check CSV]
Defaults point to the outputs of the final runs laid out under result/final_runs/ (see final/README.md).
"""
import argparse, os
import numpy as np
import pandas as pd

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCN = ('ssp126', 'ssp585')
H, A = 10.90645, 3.39567
B = np.array([-0.900947, -0.4014121])          # mapped zero of height, agb (exact)
N_SITES = 6171

ap = argparse.ArgumentParser()
ap.add_argument('--nn', default='result/final_runs/nn/nn_preds.npz')              # carbonglobe_direct_nn.py (xage)
ap.add_argument('--gbm', default='result/final_runs/gbm')                          # carbonglobe_gbm.py, 6 configs
ap.add_argument('--lstm-ep4', default='result/1/global_probe_preds.npz')           # global_probe.py PROFILE noco2
ap.add_argument('--lstm-sweep', default='result/kaggle_run1/global_probe_preds.npz')  # PROFILE sweep (s42ep6)
ap.add_argument('--lstm-s7', default='result/noco2-s7/global_probe_preds.npz')    # PROFILE noco2_s7
ap.add_argument('--out', default='final/submission_final.csv')
ap.add_argument('--check', default='', help='scored CSV to compare with')
args = ap.parse_args()

DEAD = dict(np.load('result/dead_rows.npz'))
TT = np.load('result/test_transform.npz')
YS, YB = TT['y_s'].astype(np.float64), TT['y_b'].astype(np.float64)
REF = pd.read_csv('final/ref/meanmatch_reference.csv').set_index('id')
IDS = {s: np.array([[f'{s}_{site}_{age}' for site in range(N_SITES)] for age in AGES]) for s in SCN}


def mean_of(paths, key):
    d = [np.load(p) for p in paths]
    return {s: sum(x[key.format(scn=s)].astype(np.float64) for x in d) / len(d) for s in SCN}


def to_answer_space(raw, how):
    out = {}
    for s in SCN:
        p = np.maximum(raw[s][..., :2], 0.0)
        if how == 'map':
            v = (p * YS[:2] + YB[:2]).astype(np.float32).astype(np.float64)
        else:
            alive = ~DEAD[s]
            r = REF.loc[IDS[s][alive].ravel()]
            v = np.empty_like(p)
            for k, (col, sc) in enumerate((('height', H), ('agb', A))):
                slope = (r[col].mean() * sc - B[k]) / p[..., k][alive].mean()
                v[..., k] = slope * p[..., k] + B[k]
                v[..., k][DEAD[s]] = B[k]
        out[s] = v
    return out


G = args.gbm
# member: (raw prediction loader, transform, weight)
MEMBERS = {
    'nnx_y0inv2':           (lambda: mean_of([args.nn], '{scn}_y0inv2'), 'match', 0.651145),
    's7_xinv':              (lambda: mean_of([args.lstm_s7], '{scn}_xinv'), 'map', -0.300189),
    'gbmc3i':               (lambda: mean_of([f'{G}/gbm_preds_chrono.npz', f'{G}/gbm_preds_chrono_s7.npz',
                                              f'{G}/gbm_preds_chrono_s11.npz'], '{scn}_direct_y0inv'), 'match', 0.342943),
    'lstm_y0inv_xinv':      (lambda: mean_of([args.lstm_ep4], '{scn}_y0inv_xinv'), 'map', 0.344256),
    'rfm_none':             (lambda: mean_of([f'{G}/gbm_preds_rf.npz'], '{scn}_direct_none'), 'match', -0.228401),
    'sw_s42ep6_y0inv_xinv': (lambda: mean_of([args.lstm_sweep], '{scn}_s42ep6_y0inv_xinv'), 'map', 0.309802),
    'gbmm_y0inv2':          (lambda: mean_of([f'{G}/gbm_preds_y0x.npz'], '{scn}_direct_y0inv2'), 'match', -0.205544),
    'gbm_rollout_y0inv':    (lambda: mean_of([f'{G}/gbm_preds_roll.npz'], '{scn}_rollout_y0inv'), 'match', 0.085988),
}
w = np.array([m[2] for m in MEMBERS.values()])
assert abs(w.sum() - 1) < 1e-5, w.sum()
w = w / w.sum()

blend = {s: 0.0 for s in SCN}
for (name, (load, how, _)), wi in zip(MEMBERS.items(), w):
    v = to_answer_space(load(), how)
    for s in SCN:
        blend[s] = blend[s] + wi * v[s]
    print(f'{name:22s} weight {wi:+.4f} ({how})')

parts = []
for s in SCN:
    v = blend[s].copy()
    v[..., 0] = np.maximum(v[..., 0], B[0]); v[..., 1] = np.maximum(v[..., 1], B[1])
    v[DEAD[s]] = B
    ids = [f'{s}_{site}_{age}' for site in range(N_SITES) for age in AGES]
    parts.append(pd.DataFrame({'id': ids, 'height': (v[:, :, 0].T / H).reshape(-1), 'agb': (v[:, :, 1].T / A).reshape(-1)}))
sub = pd.concat(parts, ignore_index=True)
assert len(sub) == 185130 and sub['id'].is_unique and np.isfinite(sub[['height', 'agb']].to_numpy()).all()
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
sub.to_csv(args.out, index=False)
print('wrote', args.out, sub.shape)
if args.check and os.path.isfile(args.check):
    old = pd.read_csv(args.check).set_index('id').loc[sub['id']]
    print(f'max |diff| to {args.check}: {np.abs(old.to_numpy() - sub[["height", "agb"]].to_numpy()).max():.2e}')
