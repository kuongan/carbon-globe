# %% [markdown]
# # CarbonGlobe final submission: `flatg12` (public LB 0.046)
#
# One blend layer over twelve single models, chosen by a greedy + swap search over 63 scored single models
# (INSIGHTS 4.57). Run from the repository root: `python final/make_submission_flatg12.py`.
#
# Inputs: each member's scored predictions in the answer space, `result/<member>_preds/{ssp126,ssp585}.npy`
# (submission units x scale; how each was produced: `final/README.md` and `INSIGHTS.md`), and its public LB.
# Weights in closed form: `MSE_i + MSE_j - D_ij = 2 Cov(e_i, e_j)` -> `C`, best weights summing to 1 are
# `C^-1 1 / (1' C^-1 1)` (negative weights allowed); dead rows (`result/dead_rows.npz`) exact and excluded from `C`.

# %%
import os
import numpy as np
import pandas as pd

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCN = ('ssp126', 'ssp585')
H, A = 10.90645, 3.39567
B = np.array([-0.900947, -0.4014121])
N_SITES = 6171
OUT = 'final/submission_flatg12.csv'
SUBMITTED = 'result/submission_blend_flatg12.csv'
DEAD = dict(np.load('result/dead_rows.npz'))

MEMBERS = {                       # member: public LB alone (model; script)
    'nnx_y0inv2': 0.061,            # cross-age NN, seeds 4-7 (kaggle_direct_nn_xage.ipynb)
    'lstm_y0inv_xinv': 0.109,       # LSTM-XL no-CO2 seed 42 ep4, physical y0 + forcing (global_probe.py 'noco2')
    'gbmc3i': 0.091,                # LightGBM + chronosequence + event risk, 3 seeds, y0inv (carbonglobe_gbm.py)
    'rfm_none': 0.155,              # LightGBM random-forest mode, y0 as given (carbonglobe_gbm.py GBM_BOOSTING=rf)
    'nov2_snapmean': 0.136,         # LSTM-XL no-CO2 without group v2, snapshot mean (carbonglobe-global.ipynb)
    'analog_add': 0.173,            # own-site historical growth added to test y0 (analog.py)
    's7_xinv': 0.277,               # LSTM-XL no-CO2 seed 7 ep3, physical forcing (global_probe.py 'noco2_s7')
    'sw_s42ep6_y0inv_xinv': 0.112,  # LSTM-XL seed 42 ep6, physical y0 + forcing (kaggle_sweep_pairs_run1.ipynb)
    'ecoformer_ed': 0.169,          # EcoFormer, 1-step training (early benchmark run)
    'gbmm_y0inv2': 0.097,           # LightGBM direct (no chrono), y0 past physical (carbonglobe_gbm.py)
    'gbm_rollout_y0inv': 0.237,     # LightGBM annual rollout (carbonglobe_gbm.py rollout model)
    's7_ep3': 0.118,                # LSTM-XL no-CO2 seed 7 ep3, inputs as given (carbonglobe-global.ipynb)
}

# %% members, pairwise distances, closed-form weights
P = {m: {s: np.load(f'result/{m}_preds/{s}.npy')[..., :2].astype(np.float64) for s in SCN} for m in MEMBERS}
names = list(MEMBERS); R = len(names)


def dead_err(v, s):
    m = DEAD[s]
    return 0.5 * (np.sum((v[..., 0][m] - B[0]) ** 2) / H ** 2 + np.sum((v[..., 1][m] - B[1]) ** 2) / A ** 2) / m.size


D = np.zeros((R, R))
for s in SCN:
    keep = ~DEAD[s]
    for i in range(R):
        for j in range(i + 1, R):
            dd = P[names[i]][s] - P[names[j]][s]
            D[i, j] += 0.25 * (np.sum((dd[..., 0][keep] / H) ** 2) + np.sum((dd[..., 1][keep] / A) ** 2)) / keep.size
D = D + D.T
mse = np.array([MEMBERS[m] for m in names]) - np.array([0.5 * sum(dead_err(P[m][s], s) for s in SCN) for m in names])
C = 0.5 * (mse[:, None] + mse[None, :] - D)
w = np.linalg.solve(C, np.ones(R)); w /= w.sum()
print('weights', {m: round(float(x), 4) for m, x in zip(names, w)}, f'| estimated public LB {w @ C @ w:.4f}')

# %% blend, dead rows exact, write, compare with the scored file
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
print('wrote', OUT)
if os.path.isfile(SUBMITTED):
    old = pd.read_csv(SUBMITTED).set_index('id').loc[sub['id']]
    print(f'max |diff| to the submitted {SUBMITTED}: {np.abs(old.to_numpy() - sub[["height", "agb"]].to_numpy()).max():.2e}')
