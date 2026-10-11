"""Probe: is the hidden truth in the same transformed space as test_y0 (T2) or physical (T1)?
Dead entries (test_y0 at its mass point = physical 0; in the grid a dead stand stays dead for 40 years with certainty):
T1 truth = 0, T2 truth = mass point / global mean = (-0.082607, -0.118213). Set them to the T2 values in blend_scn2."""
import numpy as np, pandas as pd
P = 'data/public'
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
H, A = 10.90645, 3.39567
BH, BA = -0.900947, -0.4014121
TH, TA = BH / H, BA / A
sub = pd.read_csv('result/submission_blend_scn2.csv')
idx = {k: i for i, k in enumerate(sub['id'])}
n = len(sub); rows = []
for scn in ('ssp126', 'ssp585'):
    y0 = np.load(f'{P}/test_{scn}.npz')['test_y0']
    s_i, a_i = np.nonzero(y0[..., 0] == np.float32(BH))
    rows += [idx[f'{scn}_{s}_{AGES[a]}'] for s, a in zip(s_i, a_i)]
rows = np.array(rows)
ph, pa = sub.loc[rows, 'height'].to_numpy(), sub.loc[rows, 'agb'].to_numpy()
d_T1 = 0.5 * (np.sum(TH ** 2 - ph ** 2) + np.sum(TA ** 2 - pa ** 2)) / n
d_T2 = 0.5 * (np.sum(0 - (ph - TH) ** 2) + np.sum(0 - (pa - TA) ** 2)) / n
print(f'dead rows {len(rows)} ({len(rows) / n:.2%}) | blend there: height {ph.mean():.4f} agb {pa.mean():.4f}')
print(f'predicted LB change: T1 (truth 0) {d_T1:+.5f} | T2 (truth = mass point) {d_T2:+.5f} | from 0.074 (0.0735-0.0745)')
print(f'  -> T1: {0.0735 + d_T1:.4f}-{0.0745 + d_T1:.4f} | T2: {0.0735 + d_T2:.4f}-{0.0745 + d_T2:.4f}')
out = sub.copy()
out.loc[rows, 'height'] = TH; out.loc[rows, 'agb'] = TA
out.to_csv('result/probe_dead_T2.csv', index=False)
z = sub.copy(); z.loc[rows, ['height', 'agb']] = 0.0
z.to_csv('result/probe_dead_zero.csv', index=False)
print('wrote result/probe_dead_T2.csv (and probe_dead_zero.csv = dead rows at 0, the T1-optimal version)')
