"""EDA part 1: growth dynamics — what does the grid simulation say about year-40 outcomes?

Key question: can we predict year-40 height / agb from test_y0 + test_x WITHOUT rolling out a model,
just using patterns in the grid simulation?

Analyses:
1. How much of year-40 variance is explained by y0 alone vs by forcing features?
2. Which forcing features best predict GROWTH (y40 - y0) in the grid?
3. Do test sites fall inside the grid's training distribution for the features that matter most?
4. Is there a biome / static-feature grouping that segments the residuals?
"""
import numpy as np, time

G = 'data/carbonglobe'; P = 'data/public'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
V = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']
CO2 = list(range(8, 32))
KEEP = [j for j in range(136) if j not in CO2 and j != 32]

t0 = time.time()
g = np.load('_grid_y.npz'); Y0 = g['Y0'][:, :, 11].astype(np.float32); Y40 = g['Y40'].astype(np.float32)
ok = np.isfinite(Y0).all(axis=(0,2)) & np.isfinite(Y40).all(axis=(0,2)) & (Y0[:,:,0].max(0) > 0)
Y0, Y40 = Y0[:, ok], Y40[:, ok]
N = ok.sum(); print(f'alive sites {N} (of {len(ok)})')

# Load a subset of forcing for those sites
from scipy.stats import pearsonr
gx = np.load(f'{G}/data_global/glob_X_fea.npy', mmap_mode='r')
FP = np.empty((N, 136), np.float32)
idx = np.flatnonzero(ok)
o = np.argsort(idx)
for i in range(0, len(o), 2048):
    sel = o[i:i + 2048]
    FP[sel] = gx[idx[sel], :, :, :].mean(axis=(1, 2))
print(f'forcing fingerprints loaded in {time.time()-t0:.0f}s')

# 1. R^2 of y0 predicting y40 per variable per age
print('\n=== 1. R^2 of y0 -> y40 per variable (alive sites) ===')
print(f'{"var":6s} age1    age50   age200  age500')
for k, vn in enumerate(V):
    row = []
    for ai in (0, 4, 8, 14):
        y0v = Y0[ai, :, k]; y40v = Y40[ai, :, k]
        m = (y0v > 0) & (y40v > 0)
        if m.sum() > 100:
            r2 = pearsonr(y0v[m], y40v[m])[0]**2
        else: r2 = float('nan')
        row.append(r2)
    print(f'{vn:6s} ' + ' '.join(f'{r:.3f}  ' for r in row))

# 2. R^2 of forcing features predicting growth (y40-y0) for height and agb
print('\n=== 2. Top features for GROWTH (y40-y0), pooled over ages 50-500 ===')
for k, vn in enumerate(['height', 'agb']):
    ages_idx = list(range(4, 15))
    g_k = np.concatenate([Y40[a, :, k] - Y0[a, :, k] for a in ages_idx])
    fp_k = np.tile(FP, (len(ages_idx), 1)).reshape(len(ages_idx)*N, 136)
    alive = np.concatenate([(Y0[a, :, 0] > 0) for a in ages_idx])
    g_k, fp_k = g_k[alive], fp_k[alive]
    rs = [(j, abs(pearsonr(fp_k[:, j], g_k)[0])) for j in KEEP if fp_k[:, j].std() > 1e-6]
    rs.sort(key=lambda x: -x[1])
    print(f'{vn}: top 10 features for growth: {[(j, round(r,3)) for j, r in rs[:10]]}')

# 3. Test sites vs grid: distribution of the top growth features
print('\n=== 3. Test distribution vs grid for top growth features ===')
tx = np.load(f'{P}/test_ssp126.npz')['test_x'][:, :, :, :].mean(axis=(1,2))
top_j = [rs[i][0] for i in range(5)]   # top 5 for agb growth
for j in top_j:
    gv = FP[:, j]; tv = tx[:, j]
    pct_in = np.mean((tv >= np.percentile(gv, 1)) & (tv <= np.percentile(gv, 99)))
    print(f'  feat {j:3d}: grid mean {gv.mean():8.3f} std {gv.std():7.3f} | test mean {tv.mean():8.3f} std {tv.std():7.3f} | test in grid p1-p99: {pct_in:.1%}')

# 4. Static feature clustering: do static features segment growth well?
print('\n=== 4. Static features (129-135) vs height growth at age 50 ===')
from numpy.polynomial.polynomial import polyfit as pfit
a50 = 4
dh50 = Y40[a50, :, 0] - Y0[a50, :, 0]
alive50 = Y0[a50, :, 0] > 0
sf = ['area','frac_tree','frac_veg','slope','roughness','cover','age_grid']
for k, sn in enumerate(sf):
    j = 129 + k
    x = FP[alive50, j]; y = dh50[alive50]
    if x.std() > 1e-6:
        r2 = pearsonr(x, y)[0]**2
        print(f'  {sn:12s} (feat {j}): R^2 vs height growth at age50 = {r2:.4f}')
