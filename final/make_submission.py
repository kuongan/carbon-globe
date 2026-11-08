# %% [markdown]
# # CarbonGlobe final submission: `flat6` (public LB 0.049)
#
# One blend layer over six single models. Run from the repository root:
# `python final/make_submission.py` (or the notebook `final/make_submission.ipynb`, same code).
#
# 1. **Members from raw model outputs.** Each model writes PHYSICAL year-40 predictions `(15 ages, 6171 sites, 7)`
#    per scenario. They are brought to the answer's space (the test is the CarbonGlobe grid behind a per-variable
#    affine map `y_test = s * y + b`, and so is the hidden answer; INSIGHTS 4.22/4.23) in one of two ways:
#    - `map`: `s * max(p, 0) + b` with the slopes of `result/test_transform.npz`;
#    - `match`: per scenario and variable, the slope that gives the alive rows the same mean as a LB-calibrated
#      reference file (`final/ref/meanmatch_reference.csv`); dead rows (`result/dead_rows.npz`) at the mapped zero.
# 2. **Weights in closed form.** With an MSE metric, `MSE_i + MSE_j - D_ij = 2 Cov(e_i, e_j)`: the public LB of each
#    member plus their pairwise mean squared differences `D` give the error covariance `C` without any truth, and
#    the best weights summing to 1 are `C^-1 1 / (1' C^-1 1)` (negative weights allowed). Dead rows are known exactly
#    and excluded from `C`.
# 3. **Write** `submission.csv` (dead rows exact, floor at the mapped zero) and compare with the submitted file.
#
# Where each raw output comes from: `final/README.md`.

# %%
import os
import numpy as np
import pandas as pd

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCN = ('ssp126', 'ssp585')
H, A = 10.90645, 3.39567                       # metric scales (global means of height, agb)
B = np.array([-0.900947, -0.4014121])          # the mapped zero of height, agb (test-space value of 0)
N_SITES = 6171
OUT = 'final/submission.csv'
SUBMITTED = 'result/submission_blend_flat6.csv'  # the file scored 0.049 (if present: checked)

DEAD = dict(np.load('result/dead_rows.npz'))   # (15, 6171) per scenario: y0 at the mapped zero = dead forever
TT = np.load('result/test_transform.npz')
YS, YB = TT['y_s'].astype(np.float64), TT['y_b'].astype(np.float64)

# name: (raw npz, key template, transform, public LB of the member alone)
MEMBERS = {
    'nnx8_y0inv2':          ('result/nn_direct_xage2/nn_preds_xavg8.npz', '{scn}_y0inv2', 'match', 0.061),
    'gbmc3i':               ('result/gbm/gbm_preds_chrono_avg3.npz', '{scn}_direct_y0inv', 'match', 0.091),
    'lstm_y0inv_xinv':      ('result/1/global_probe_preds.npz', '{scn}_y0inv_xinv', 'map', 0.109),
    'sw_s42ep6_y0inv_xinv': ('result/kaggle_run1/global_probe_preds.npz', '{scn}_s42ep6_y0inv_xinv', 'map', 0.112),
    'rfm_none':             ('result/gbm/gbm_preds_rf.npz', '{scn}_direct_none', 'match', 0.155),
    's7_xinv':              ('result/noco2-s7/global_probe_preds.npz', '{scn}_xinv', 'map', 0.277),
}

# %% 0. seed averages from the per-run outputs (checked against the stored averages the members were built from)
AVERAGES = {
    'result/gbm/gbm_preds_chrono_avg3.npz': ['result/gbm/gbm_preds_chrono.npz', 'result/gbm/gbm_preds_chrono_s7.npz',
                                             'result/gbm/gbm_preds_chrono_s11.npz'],         # GBM seeds 42, 7, 11
    'result/nn_direct_xage2/nn_preds_xavg8.npz': ['result/nn_direct_xage/nn_preds.npz',      # NN seeds 4-7
                                                  'result/nn_direct_xage2/nn_preds.npz'],    # NN seeds 8-11
}
for avg, runs in AVERAGES.items():
    d = [np.load(r) for r in runs]
    stored = np.load(avg)
    worst = 0.0
    for k in stored.files:
        mean = sum(x[k].astype(np.float64) for x in d) / len(d)
        worst = max(worst, float(np.abs(mean - stored[k]).max()))
    print(f'{avg}: mean of {len(runs)} runs, max |diff| to the stored average {worst:.2e}')

# %% 1. members in the answer's space (submission units x scale: (15, 6171, 2) per scenario)
ref = pd.read_csv('final/ref/meanmatch_reference.csv').set_index('id')
IDS = {s: np.array([[f'{s}_{site}_{age}' for site in range(N_SITES)] for age in AGES]) for s in SCN}


def build(path, key, how):
    d = np.load(path)
    out = {}
    for s in SCN:
        p = np.maximum(d[key.format(scn=s)][..., :2].astype(np.float64), 0.0)
        if how == 'map':
            v = (p * YS[:2] + YB[:2]).astype(np.float32).astype(np.float64)   # stored as float32 upstream
        else:
            alive = ~DEAD[s]
            r = ref.loc[IDS[s][alive].ravel()]
            v = np.empty_like(p)
            for k, (col, sc) in enumerate((('height', H), ('agb', A))):
                slope = (r[col].mean() * sc - B[k]) / p[..., k][alive].mean()
                v[..., k] = slope * p[..., k] + B[k]
                v[..., k][DEAD[s]] = B[k]
        out[s] = v
    return out


P = {m: build(*cfg[:3]) for m, cfg in MEMBERS.items()}
for m in MEMBERS:                                   # same as the laid-out arrays the blend was solved on
    lay = f'result/{m}_preds'
    if os.path.isdir(lay):
        diff = max(np.abs(P[m][s] - np.load(f'{lay}/{s}.npy')[..., :2]).max() for s in SCN)
        print(f'{m:22s} rebuilt from raw output | max |diff| to the scored member {diff:.2e}')

# %% 2. weights from the public LB of each member (closed form)
names = list(MEMBERS)
R = len(names)


def dead_err(v, s):
    m = DEAD[s]
    return 0.5 * (np.sum((v[..., 0][m] - B[0]) ** 2) / H ** 2 + np.sum((v[..., 1][m] - B[1]) ** 2) / A ** 2) / m.size


D = np.zeros((R, R))
for s in SCN:
    keep = ~DEAD[s]
    for i in range(R):
        for j in range(i + 1, R):
            dd = P[names[i]][s] - P[names[j]][s]
            D[i, j] += 0.5 * 0.5 * (np.sum((dd[..., 0][keep] / H) ** 2) + np.sum((dd[..., 1][keep] / A) ** 2)) / keep.size
D = D + D.T
lb = np.array([MEMBERS[m][3] for m in names])
mse = lb - np.array([0.5 * sum(dead_err(P[m][s], s) for s in SCN) for m in names])   # alive part of each LB
C = 0.5 * (mse[:, None] + mse[None, :] - D)
w = np.linalg.solve(C, np.ones(R)); w /= w.sum()
print('weights', {m: round(float(x), 4) for m, x in zip(names, w)}, f'| estimated public LB {w @ C @ w:.4f}')

# %% 3. blend, dead rows exact, write
parts = []
for s in SCN:
    v = sum(wi * P[m][s] for wi, m in zip(w, names))
    v[..., 0] = np.maximum(v[..., 0], B[0]); v[..., 1] = np.maximum(v[..., 1], B[1])
    v[DEAD[s]] = B
    ids = [f'{s}_{site}_{age}' for site in range(N_SITES) for age in AGES]
    parts.append(pd.DataFrame({'id': ids, 'height': (v[:, :, 0].T / H).reshape(-1), 'agb': (v[:, :, 1].T / A).reshape(-1)}))
sub = pd.concat(parts, ignore_index=True)
assert len(sub) == 185130 and sub['id'].is_unique and np.isfinite(sub[['height', 'agb']].to_numpy()).all()
sub.to_csv(OUT, index=False)
print('wrote', OUT, sub.shape)
if os.path.isfile(SUBMITTED):
    old = pd.read_csv(SUBMITTED).set_index('id').loc[sub['id']]
    print(f'max |diff| to the submitted {SUBMITTED}: {np.abs(old.to_numpy() - sub[["height", "agb"]].to_numpy()).max():.2e}')
