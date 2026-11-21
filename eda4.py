"""EDA part 4: CO2 response signal — the key gap between LSTM and the truth.

The test uses future CO2 (ssp126/ssp585 SSP paths). The LSTM was trained without CO2 (CO2_MODE=const).
What does the grid simulation tell us about the CO2 fertilization effect on year-40 outcomes?
Can we build a post-hoc CO2 correction?
"""
import numpy as np, time
from scipy.stats import pearsonr

G = 'data/carbonglobe'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]

t0 = time.time()
gz = np.load('_grid_y.npz')
Y0 = gz['Y0'][:, :, 11].astype(np.float32); Y40 = gz['Y40'].astype(np.float32)
ok = np.isfinite(Y0).all(axis=(0,2)) & np.isfinite(Y40).all(axis=(0,2)); N = ok.sum()
Y0, Y40 = Y0[:, ok], Y40[:, ok]
gx = np.load(f'{G}/data_global/glob_X_fea.npy', mmap_mode='r')
CO2 = np.empty(N, np.float32)
idx = np.flatnonzero(ok); o = np.argsort(idx)
for i in range(0, len(o), 2048):
    sel = o[i:i+2048]; CO2[sel] = gx[idx[sel], :, :, 8].mean()  # 40-yr mean of CO2 feature 8
print(f'loaded in {time.time()-t0:.0f}s | CO2 range [{CO2.min():.1f}, {CO2.max():.1f}] mean {CO2.mean():.1f}')

# 1. CO2 level effect on growth, controlling for y0
print('\n=== 1. Partial R^2 of CO2 on growth (age 50, alive sites) ===')
a50 = 4; alive = Y0[a50, :, 0] > 0
for k, vn in enumerate(['height', 'agb']):
    grow = Y40[a50, :, k] - Y0[a50, :, k]
    y0 = Y0[a50, :, k]
    # residual of grow after removing y0
    res = grow[alive] - np.polyval(np.polyfit(y0[alive], grow[alive], 1), y0[alive])
    r_co2 = pearsonr(CO2[alive], res)[0]
    print(f'  {vn}: R(grow, y0) = {pearsonr(y0[alive], grow[alive])[0]:.3f}  R(partial CO2) = {r_co2:.3f}  R^2_partial = {r_co2**2:.4f}')

# 2. CO2 response curve: growth vs CO2 in quartiles
print('\n=== 2. Growth by CO2 quartile (age 50, height, alive) ===')
co2_q = np.percentile(CO2[alive], [0,25,50,75,100])
grow_h = Y40[a50,:,0][alive] - Y0[a50,:,0][alive]
for lo, hi in zip(co2_q[:-1], co2_q[1:]):
    m = (CO2[alive] >= lo) & (CO2[alive] < hi)
    print(f'  CO2 [{lo:.0f},{hi:.0f}]: n={m.sum()} mean_growth={grow_h[m].mean():.3f} mean_y40={Y40[a50,:,0][alive][m].mean():.3f} mean_y0={Y0[a50,:,0][alive][m].mean():.3f}')

# 3. Test CO2 levels vs grid range
print('\n=== 3. Test CO2 levels vs grid distribution ===')
from scipy.stats import percentileofscore
for scn in ('ssp126', 'ssp585'):
    tx_co2 = np.load(f'data/public/test_{scn}.npz')['test_x'][:, :, :, 8].mean()
    pct = percentileofscore(CO2, tx_co2)
    print(f'  {scn} mean CO2 {tx_co2:.1f} -> grid percentile {pct:.1f}%')
print(f'  Grid mean {CO2.mean():.1f} | test is lower: the model saw higher CO2 in training')

# 4. Counterfactual: if we scale y40 by a CO2 factor, how much does it help?
# Estimate: how much does y40 increase per unit CO2 increase, HOLDING y0 fixed?
print('\n=== 4. CO2 sensitivity: d(y40)/d(CO2) normalized ===')
for k, vn in enumerate(['height', 'agb']):
    betas = []
    for a in range(4, 15):   # ages 50-500
        alive_a = Y0[a, :, 0] > 0
        y40v = Y40[a, :, k][alive_a]
        co2v = CO2[alive_a]
        y0v = Y0[a, :, k][alive_a]
        # multiple regression: y40 ~ b0*y0 + b1*CO2 + const
        X = np.stack([y0v, co2v, np.ones_like(co2v)], 1)
        beta, _, _, _ = np.linalg.lstsq(X, y40v, rcond=None)
        betas.append(beta[1])   # d(y40)/d(CO2), controlling for y0
    print(f'  {vn}: d(y40)/d(CO2) ages 50-500 mean {np.mean(betas):.4f} std {np.std(betas):.4f}')
    # scale: if CO2 increases by (ssp585_mean - ssp126_mean) ppm, by how much does y40 change?
    co2_diff = 7.63   # from eda2.py
    print(f'    Predicted y40 increase for +{co2_diff:.1f} ppm CO2: {np.mean(betas)*co2_diff:.4f} ({np.mean(betas)*co2_diff/13.0*100:.2f}% of typical y40 height)')

# 5. Can we identify a CO2 correction for the LSTM's blind spot?
print('\n=== 5. Predicted CO2 correction for no-CO2 LSTM (ssp585 vs ssp126 gap) ===')
# The LSTM predicts the same output for both scenarios (CO2 is zeroed). The truth differs by CO2.
# For the grid: training used 1980-2020 CO2 (mean ~365 ppm). Test CO2: ssp126 ~370, ssp585 ~378.
co2_grid_mean = CO2.mean()   # ~365
co2_126_test = 370.05; co2_585_test = 377.67
for k, vn in enumerate(['height', 'agb']):
    betas = []
    for a in range(4, 15):
        alive_a = Y0[a, :, 0] > 0
        y40v = Y40[a, :, k][alive_a]; co2v = CO2[alive_a]; y0v = Y0[a, :, k][alive_a]
        X = np.stack([y0v, co2v, np.ones_like(co2v)], 1)
        beta, _, _, _ = np.linalg.lstsq(X, y40v, rcond=None)
        betas.append(beta[1])
    b = np.mean(betas)
    adj_126 = b * (co2_126_test - co2_grid_mean)
    adj_585 = b * (co2_585_test - co2_grid_mean)
    print(f'  {vn}: grid mean CO2 {co2_grid_mean:.1f} -> adj ssp126 {adj_126:+.4f}  adj ssp585 {adj_585:+.4f} (physical units)')
    # In submission units (/ global mean):
    sc = 10.90645 if k == 0 else 3.39567
    print(f'    Submission scale (/mean): ssp126 {adj_126/sc:+.5f}  ssp585 {adj_585/sc:+.5f}')
