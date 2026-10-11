"""Persistence baseline: predict the year-40 state = the year-0 state (test_y0). CPU, a few seconds.

Paste into one Kaggle notebook cell (no GPU, no repo clone). Input: the competition data only.
Writes /kaggle/working/submission_y0.csv (submit it once) and /kaggle/working/y0_preds.npz (send it back).

Why: as a blend component, y0 lets the blend move each scenario between "level" and "growth"
(y40' = sum_i w_i p_i + w_0 y0). The LB oracle needs one score of this file (+ one hybrid) to use it.
"""
import numpy as np
import pandas as pd

COMP = '/kaggle/input/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public'
OUT = '/kaggle/working'
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
H_SCALE, A_SCALE = 10.90645, 3.39567

sample_sub = pd.read_csv(f'{COMP}/sample_submission.csv')
parts, arrs = [], {}
for scn in ('ssp126', 'ssp585'):
    y0 = np.load(f'{COMP}/test_{scn}.npz')['test_y0'].astype(np.float32)    # (N sites, 15 ages, 7)
    p = np.maximum(np.transpose(y0, (1, 0, 2)), 0.0)                      # (15, N, 7), physical units
    arrs[scn] = p
    N = p.shape[1]
    parts.append(pd.DataFrame({'id': [f'{scn}_{s}_{a}' for a in SEED_AGES for s in range(N)],
                               'height': (p[:, :, 0] / H_SCALE).reshape(-1),
                               'agb': (p[:, :, 1] / A_SCALE).reshape(-1)}))
    print(f'{scn}: y0 scaled means height {p[..., 0].mean() / H_SCALE:.3f} agb {p[..., 1].mean() / A_SCALE:.3f}')
sub = sample_sub[['id']].join(pd.concat(parts).set_index('id'), on='id')
assert len(sub) == len(sample_sub) and not sub[['height', 'agb']].isna().any().any()
sub.to_csv(f'{OUT}/submission_y0.csv', index=False)
np.savez_compressed(f'{OUT}/y0_preds.npz', **arrs)
print('wrote submission_y0.csv and y0_preds.npz')
