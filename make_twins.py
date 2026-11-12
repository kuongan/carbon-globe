# %% [markdown]
# # Twin members of the two neural models (CPU, ~2 min)
#
# The cross-age NN (`nnx8`, LB 0.061) and the chrono NN (`nnc`, LB 0.075) entered the blend with their `y0inv2`
# variant only. Their `y0inv` twins (y0 mapped back to physical, not pushed past it) were never scored; ranked
# against the blend they are the steepest remaining candidates (`rank_members.py`: a miss of 0.01 vs their predicted
# LB moves the blend to ~0.047). Same mean matching as every other member (`slope.py match`).
#
# Input: one Kaggle dataset made from `kaggle_upload/twins/` (nn_xage_preds.npz, nn_xage2_preds.npz,
# nn_chrono_preds.npz, meanmatch_reference.csv, dead_rows.npz). Output (/kaggle/working):
# `match_nnx8i.csv`, `match_nnc_y0inv.csv` -> submit both, send the two LB scores.

# %%
import glob, os
import numpy as np
import pandas as pd

hits = sorted(glob.glob('/kaggle/input/**/meanmatch_reference.csv', recursive=True))
IN = os.path.dirname(hits[0]) if hits else 'kaggle_upload/twins'
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else 'result'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCN = ('ssp126', 'ssp585')
H, A = 10.90645, 3.39567
B = {'height': -0.900947, 'agb': -0.4014121}
print('inputs from', IN)

ref = pd.read_csv(f'{IN}/meanmatch_reference.csv').set_index('id')
dead = np.load(f'{IN}/dead_rows.npz')


def match(preds, path):
    """per scenario and variable: slope giving the alive rows the reference mean; dead rows at the mapped zero"""
    parts = []
    for scn in SCN:
        p = np.maximum(preds[scn][..., :2].astype(np.float64), 0.0); N = p.shape[1]; alive = ~dead[scn]
        ids = np.array([[f'{scn}_{site}_{age}' for site in range(N)] for age in AGES])
        r = ref.loc[ids[alive].ravel()]
        out = {}
        for k, (col, sc) in enumerate((('height', H), ('agb', A))):
            s = (r[col].mean() * sc - B[col]) / p[..., k][alive].mean()
            v = s * p[..., k] + B[col]; v[dead[scn]] = B[col]
            out[col] = v / sc
            print(f'  {scn} {col}: slope {s:.4f}')
        parts.append(pd.DataFrame({'id': ids.T.reshape(-1), 'height': out['height'].T.reshape(-1),
                                   'agb': out['agb'].T.reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)
    assert len(sub) == 185130 and sub.id.is_unique and np.isfinite(sub[['height', 'agb']].to_numpy()).all()
    sub.to_csv(path, index=False)
    print('wrote', path)


# %% cross-age NN, 8 seeds (4-7 and 8-11), y0inv variant
a, b = np.load(f'{IN}/nn_xage_preds.npz'), np.load(f'{IN}/nn_xage2_preds.npz')
match({s: (a[f'{s}_y0inv'].astype(np.float64) + b[f'{s}_y0inv']) / 2 for s in SCN}, f'{OUT}/match_nnx8i.csv')

# %% chrono NN, 4 seeds, y0inv variant
c = np.load(f'{IN}/nn_chrono_preds.npz')
match({s: c[f'{s}_y0inv'] for s in SCN}, f'{OUT}/match_nnc_y0inv.csv')

# %% check: the y0inv2 member rebuilt the same way must equal the scored nnx8 member's slopes (0.757/0.794/0.783/0.863)
match({s: (a[f'{s}_y0inv2'].astype(np.float64) + b[f'{s}_y0inv2']) / 2 for s in SCN}, f'{OUT}/check_nnx8_y0inv2.csv')
