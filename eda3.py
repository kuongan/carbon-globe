"""EDA part 3: residual analysis — what the LSTM model gets wrong.

Compare best predictions (probe_dead_T2.csv) to grid truth (matched sites):
- Error breakdown by age, biome (static features), and y0 level
- Are errors systematic or random?
- What pattern do the GBM trees find that LSTM misses?
"""
import numpy as np, pandas as pd
from scipy.stats import pearsonr

P = 'data/public'; G = 'data/carbonglobe'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
H, A = 10.90645, 3.39567
V = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']

z = np.load('result/test_transform.npz')
J, by, sy = z['test_to_grid'], z['y_b'], z['y_s']
good = np.flatnonzero(J >= 0); Jg = J[good]

gz = np.load('_grid_y.npz')
Y0g = gz['Y0'][:, :, 11].astype(np.float64)   # (15, 54152, 7)
Y40g = gz['Y40'].astype(np.float64)            # (15, 54152, 7)

# matched-site truth (physical): y40 of the matched grid site at the same age
# The test runs start from a different y0, so this is NOT the true y40 for the test site
# but it tells us what the sim predicts for that site under historical forcing
y0_matched = Y0g[:, Jg]                        # (15, N_good, 7)
y40_matched = Y40g[:, Jg]                      # ground truth for the matched grid site

# Test y0 in physical units (height exact, others estimated)
ty = np.transpose(np.load(f'{P}/test_ssp126.npz')['test_y0'].astype(np.float64), (1,0,2))  # (15, N, 7)
ty_phys = np.maximum((ty - by) / sy, 0.0)

# Best blend predictions in physical units (mapped back)
sub = pd.read_csv('result/probe_dead_T2.csv').set_index('id')
pred = np.zeros((2, 15, 6171, 2))  # (scn, age, site, var)
for si, scn in enumerate(('ssp126', 'ssp585')):
    for ai, age in enumerate(AGES):
        ids = [f'{scn}_{s}_{age}' for s in range(6171)]
        v = sub.loc[ids]
        pred[si, ai, :, 0] = v['height'].to_numpy() * H
        pred[si, ai, :, 1] = v['agb'].to_numpy() * A

# Back to physical from test space (used T2 convention: pred_T = s * pred_phys + b)
# But probe_dead_T2 is blend_scn2 predictions which were ALREADY in the test/truth space
# So: truth ~ pred + error; pred_phys ~ (pred_T - b) / s
pred_phys = np.maximum((pred - by[:2]) / sy[:2], 0.0)  # (2, 15, 6171, 2)

# 1. Error by age: plot proxy — how does residual scale with age?
print('=== 1. Blend prediction stats by age (ssp126, height, alive sites) ===')
dead_mask = ty[0, :, 0] < 1e-3   # approx dead in physical space
alive = ~dead_mask
print(f'{"age":6s} pred_mean  blend_mean  y0_mean  y0_std  n_alive')
for ai, age in enumerate(AGES):
    p = pred_phys[0, ai, :, 0][alive]
    y0 = ty_phys[ai, :, 0][alive]
    print(f'{age:6d} {p.mean():9.3f}  {p.mean():10.3f}  {y0.mean():7.3f}  {y0.std():6.3f}  {alive.sum()}')

# 2. Residual (blend - grid_truth for matched sites) as a function of y0
print('\n=== 2. Blend vs matched-grid-truth at age 100 and 300, height (alive, matched) ===')
for ai, age in enumerate([6, 11]):   # age100, age300
    print(f'\nAge {AGES[ai]}:')
    msite = good[Y0g[ai, Jg, 0] > 0]  # matched, alive in grid
    g40h = y40_matched[ai, np.where(np.isin(good, msite))[0], 0]
    p_h = pred_phys[0, ai, msite, 0]
    err = p_h - g40h
    y0_p = ty_phys[ai, msite, 0]
    # binned by y0
    bins = np.percentile(y0_p, [0, 20, 40, 60, 80, 100])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y0_p >= lo) & (y0_p < hi)
        if m.sum() < 5: continue
        print(f'  y0 [{lo:.1f},{hi:.1f}]: n={m.sum():4d} mean_err={err[m].mean():+.3f} std_err={err[m].std():.3f}')

# 3. Do GBM residuals correlate with y0 level? (GBM pattern vs LSTM)
print('\n=== 3. GBM avg vs LSTM: residual correlation with y0 level ===')
gbm_sub = pd.read_csv('result/match_gbm_avg_y0inv.csv').set_index('id')
for si, scn in enumerate(('ssp126',)):
    for ai in (4, 6, 10):
        age = AGES[ai]
        ids = [f'{scn}_{s}_{age}' for s in range(6171)]
        p_lstm = sub.loc[ids, 'height'].to_numpy() * H
        p_gbm = gbm_sub.loc[ids, 'height'].to_numpy() * H
        y0_h = ty_phys[ai, :, 0]
        diff = p_gbm - p_lstm
        r_diff_y0 = pearsonr(y0_h[alive], diff[alive])[0]
        r_lstm_y0 = pearsonr(y0_h[alive], p_lstm[alive])[0]
        r_gbm_y0 = pearsonr(y0_h[alive], p_gbm[alive])[0]
        print(f'  age {age}: corr(GBM-LSTM, y0) = {r_diff_y0:+.3f}  corr(LSTM,y0) = {r_lstm_y0:.3f}  corr(GBM,y0) = {r_gbm_y0:.3f}  mean_diff {diff.mean():+.3f}')

# 4. Scenario sensitivity: GBM vs LSTM response to ssp585 vs ssp126
print('\n=== 4. Scenario sensitivity (ssp585-ssp126) gap in prediction ===')
for ai in (4, 6, 10):
    age = AGES[ai]
    for name, source in (('LSTM', sub), ('GBM', gbm_sub)):
        ids126 = [f'ssp126_{s}_{age}' for s in range(6171)]
        ids585 = [f'ssp585_{s}_{age}' for s in range(6171)]
        h126 = source.loc[ids126, 'height'].mean()
        h585 = source.loc[ids585, 'height'].mean()
        a126 = source.loc[ids126, 'agb'].mean()
        a585 = source.loc[ids585, 'agb'].mean()
        print(f'  age {age} {name}: height gap {h585-h126:+.4f}  agb gap {a585-a126:+.4f}')
