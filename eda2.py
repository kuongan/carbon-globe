"""EDA part 2: feature groups v1-v4 identity, CO2 effect, scenario gap in test."""
import numpy as np, time
from scipy.stats import pearsonr

G = 'data/carbonglobe'; P = 'data/public'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
V = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']

g = np.load('_grid_y.npz'); Y0 = g['Y0'][:, :, 11].astype(np.float32); Y40 = g['Y40'].astype(np.float32)
ok = np.isfinite(Y0).all(axis=(0,2)) & np.isfinite(Y40).all(axis=(0,2))
Y0, Y40 = Y0[:, ok], Y40[:, ok]; N = ok.sum()
gx = np.load(f'{G}/data_global/glob_X_fea.npy', mmap_mode='r')
FP = np.empty((N, 136), np.float32)
idx = np.flatnonzero(ok); o = np.argsort(idx)
for i in range(0, len(o), 2048):
    sel = o[i:i+2048]; FP[sel] = gx[idx[sel]].mean(axis=(1,2))

# --- What are features v1-v4 (33+4k, 34+4k, 35+4k, 36+4k)? ---
print('=== Feature group identities v1-v4 (annual mean over the grid) ===')
print('Feature structure: 96 features = 24 vertical layers x 4 variables (v1-v4)')
print('v1=33,37,..125  v2=34,38,..126  v3=35,39,..127  v4=36,40,..128')
# v4 was identified as climatology (no interannual variation). Check v1-v3 seasonality vs v4.
gx_sub = np.array(gx[idx[:500]], np.float64)  # (500, 40, 12, 136)
v1_std = gx_sub[:, :, :, 33::4].std(axis=2).mean()   # std over months -> seasonality
v2_std = gx_sub[:, :, :, 34::4].std(axis=2).mean()
v3_std = gx_sub[:, :, :, 35::4].std(axis=2).mean()
v4_std = gx_sub[:, :, :, 36::4].std(axis=2).mean()
print(f'  Seasonal std: v1 {v1_std:.5f}  v2 {v2_std:.5f}  v3 {v3_std:.5f}  v4 {v4_std:.5f}')
v1_ys = gx_sub[:, :, :, 33::4].std(axis=1).mean()   # std over years -> interannual
v2_ys = gx_sub[:, :, :, 34::4].std(axis=1).mean()
v3_ys = gx_sub[:, :, :, 35::4].std(axis=1).mean()
v4_ys = gx_sub[:, :, :, 36::4].std(axis=1).mean()
print(f'  Interannual std: v1 {v1_ys:.5f}  v2 {v2_ys:.5f}  v3 {v3_ys:.5f}  v4 {v4_ys:.5f}')
# Correlation among layers within each group (are all 24 layers the same variable at diff depths?)
for gname, js in (('v1', range(33, 129, 4)), ('v2', range(34, 129, 4))):
    js = list(js)
    r = np.corrcoef(FP[:, js].T)
    print(f'  {gname}: median corr between layers {np.median(r[np.triu_indices_from(r,1)]):.3f}')

# --- CO2 effect on growth in the grid ---
print('\n=== CO2 effect on growth ===')
co2_mean = FP[:, 8]   # all 24 CO2 features are near-identical (checked); use feat 8
for k, vn in enumerate(['height', 'agb']):
    a50 = 4; grow = Y40[a50, :, k] - Y0[a50, :, k]; alive = Y0[a50, :, 0] > 0
    r2 = pearsonr(co2_mean[alive], grow[alive])[0]**2
    print(f'  R^2 CO2 level vs {vn} growth at age50: {r2:.4f}')

# --- Scenario gap in test ---
print('\n=== Scenario gap in test (ssp585 - ssp126 at year 40) ===')
d126 = np.load(f'{P}/test_ssp126.npz')
d585 = np.load(f'{P}/test_ssp585.npz')
# test forcing: how much does CO2 differ between scenarios?
co2_126 = d126['test_x'][:, :, :, 8].mean()
co2_585 = d585['test_x'][:, :, :, 8].mean()
print(f'  CO2 mean ssp126 {co2_126:.2f}  ssp585 {co2_585:.2f}  diff {co2_585-co2_126:.2f}')
# Current blend's scenario gap
import pandas as pd
sub = pd.read_csv('result/probe_dead_T2.csv').set_index('id')
for vn in ('height', 'agb'):
    m126 = sub[sub.index.str.startswith('ssp126')][vn].mean()
    m585 = sub[sub.index.str.startswith('ssp585')][vn].mean()
    print(f'  Blend {vn}: ssp126 {m126:.4f}  ssp585 {m585:.4f}  gap {m585-m126:+.4f}')

# --- y0 distribution: does grid span the test range? ---
print('\n=== y0 test range vs grid range (physical units, age 50) ===')
m = np.load('result/test_transform.npz'); by = m['y_b']; sy = m['y_s']
for k, vn in enumerate(V[:3]):
    ty = d126['test_y0'][:, 4, k].astype(np.float64)   # age index 4 = age 50
    ty_phys = np.maximum((ty - float(by[k])) / float(sy[k]), 0)
    gy = Y0[4, :, k]
    print(f'  {vn}: test phys p5/p50/p95 [{np.percentile(ty_phys,5):.2f}, {np.percentile(ty_phys,50):.2f}, {np.percentile(ty_phys,95):.2f}] | grid [{np.percentile(gy,5):.2f}, {np.percentile(gy,50):.2f}, {np.percentile(gy,95):.2f}]')
