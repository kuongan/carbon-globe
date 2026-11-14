"""Builds _matched_traj.npz for analog.py (final blend member analog_add) from the raw data. Local CPU, a few minutes.

The test sites are grid sites behind affine maps of the static features (test_transform.py). Same matching as there
(quantile fit, then 5 rounds of refitting the 7 static maps on the closest half of the nearest pairs), keeping the
5989 test sites closest to a grid site (distance <= 1.01e-6 in standardized units: the 5988 exact matches + one at
1.004e-6). Writes good (test site indices), J (their grid indices) and Yt (15 ages, n, 41 years, 7) of those grid sites.
usage: python analog_build.py
"""
import numpy as np
from scipy.spatial import cKDTree

G, P = 'data/carbonglobe', 'data/public'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
N_GOOD = 5989

gx = np.load(f'{G}/data_global/glob_X_fea.npy', mmap_mode='r')
S = np.concatenate([np.asarray(gx[i:i + 4096, 0, 0, 129:136], np.float64) for i in range(0, gx.shape[0], 4096)])
okg = np.flatnonzero(np.isfinite(S).all(1))
T = np.load(f'{P}/test_ssp126.npz')['test_x'][:, 0, 0, 129:136].astype(np.float64)
ps = np.linspace(0.02, 0.98, 49)
fit = np.array([np.polyfit(np.quantile(S[okg, j], ps), np.quantile(T[:, j], ps), 1) for j in range(7)])
sd = S[okg].std(0); tree = cKDTree(S[okg] / sd)
for it in range(5):
    d, j = tree.query(((T - fit[:, 1]) / fit[:, 0]) / sd)
    close = d < np.quantile(d, 0.5)
    fit = np.array([np.polyfit(S[okg[j[close]], k], T[close, k], 1) for k in range(7)])
d, j = tree.query(((T - fit[:, 1]) / fit[:, 0]) / sd)
good = np.sort(np.argsort(d)[:N_GOOD])
J = okg[j[good]]
print(f'matched {len(good)} test sites, max distance {d[good].max():.3e}, distinct grid sites {len(np.unique(J))}')
o = np.argsort(J)
Yt = np.empty((len(AGES), len(J), 41, 7), np.float32)
for ai, age in enumerate(AGES):
    g = np.load(f'{G}/data_global/glob_Y{age:03d}.npy', mmap_mode='r')
    Yt[ai, o] = np.asarray(g[J[o]][:, :, 11, :], np.float32)
np.savez_compressed('_matched_traj.npz', good=good, J=J, Yt=Yt)
print('wrote _matched_traj.npz', Yt.shape)
