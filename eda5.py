"""EDA part 5: build CO2 correction probe and test-space-aware correction.

Insight from eda4:
- Grid CO2 range: 377-396 ppm (mean 385). Test: ssp126=370, ssp585=378 (BELOW the grid!)
- Both test scenarios are in the LOWER TAIL of grid CO2 (p0 and p2)
- d(y40_height)/d(CO2) = +0.091 (more CO2 -> more growth)
- Correction: LSTM was trained at grid CO2 (~385). Test CO2 is lower -> LSTM OVER-predicts height
  adj_height_ssp126 = 0.091 * (370 - 385) = -1.37  (about -0.125 in submission units)
  adj_height_ssp585 = 0.091 * (378 - 385) = -0.64  (about -0.059 in submission units)
- For agb the effect is negative and near zero.

This is a new type of correction: the LSTM sees CO2=0 in the test (no-CO2 mode), but the truth
was generated with future CO2. The CO2 level in the grid during training was ~385 ppm.
The truth should be based on simulating at the LOWER test CO2 -> lower growth than the LSTM predicts.

Probe file: apply the per-scenario CO2 adjustment to probe_dead_T2.csv.
This changes the overall level, not the spatial pattern. Expected effect: ~2-3% reduction in height.
"""
import numpy as np, pandas as pd

P = 'data/public'
H, A = 10.90645, 3.39567
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]

sub = pd.read_csv('result/probe_dead_T2.csv').set_index('id')
out = sub.copy()

# CO2 levels (raw ppm)
co2_grid_mean = 385.4
co2_levels = {'ssp126': 370.05, 'ssp585': 377.67}

# d(y40_phys)/d(CO2) from eda4:
beta_h = 0.0913   # height
beta_a = -0.0126  # agb (negative, very small)

# Dead entries: don't correct (truth = mapped zero, known exactly)
dead = np.load('result/dead_rows.npz')
B_H, B_A = -0.900947, -0.4014121

print('CO2 corrections to apply (submission units = physical/global_mean):')
for scn in ('ssp126', 'ssp585'):
    dco2 = co2_levels[scn] - co2_grid_mean
    adj_h = beta_h * dco2 / H  # in submission units
    adj_a = beta_a * dco2 / A
    print(f'  {scn}: dCO2 = {dco2:+.1f} ppm -> height {adj_h:+.5f}  agb {adj_a:+.5f} (submission units)')
    # apply to all rows of this scenario
    mask = sub.index.str.startswith(scn)
    # apply: but don't touch dead rows (they're exact)
    d_mask = dead[scn]   # (15, 6171) boolean, axes (age_idx, site)
    a_i, s_i = np.nonzero(d_mask)
    dead_ids = {f'{scn}_{s}_{AGES[a]}' for s, a in zip(s_i, a_i)}
    alive_ids = [i for i in sub.index[mask] if i not in dead_ids]
    out.loc[alive_ids, 'height'] = sub.loc[alive_ids, 'height'] + adj_h
    out.loc[alive_ids, 'agb'] = sub.loc[alive_ids, 'agb'] + adj_a

out.reset_index(inplace=False).to_csv('result/probe_co2_adj.csv', index=False)
# compute expected change in LB (2-way blend formula: this is a shift, not a new member)
ref_h = sub['height'].mean(); new_h = out['height'].mean()
ref_a = sub['agb'].mean(); new_a = out['agb'].mean()
print(f'\nProbe stats vs original:')
print(f'  height mean: {ref_h:.5f} -> {new_h:.5f} (change {new_h-ref_h:+.5f})')
print(f'  agb mean:    {ref_a:.5f} -> {new_a:.5f} (change {new_a-ref_a:+.5f})')

# msd to the original
msd = 0.5*(((out['height']-sub['height'])**2).mean() + ((out['agb']-sub['agb'])**2).mean())
print(f'  msd to probe_dead_T2.csv: {msd:.5f}')
print('\nwrote result/probe_co2_adj.csv')
print('Submit this; if LB < 0.072, the CO2 adjustment direction is correct.')
print('Then submit result/probe_co2_adj_strong.csv with 2x the adjustment to find the optimum slope.')

# Also make a 2x version for the solve step
out2 = sub.copy()
for scn in ('ssp126', 'ssp585'):
    dco2 = co2_levels[scn] - co2_grid_mean
    adj_h = 2 * beta_h * dco2 / H
    adj_a = 2 * beta_a * dco2 / A
    mask = sub.index.str.startswith(scn)
    d_mask = dead[scn]; s_i, a_i = np.nonzero(d_mask)
    dead_ids = {f'{scn}_{s}_{AGES[a]}' for s, a in zip(s_i, a_i)}
    alive_ids = [i for i in sub.index[mask] if i not in dead_ids]
    out2.loc[alive_ids, 'height'] = sub.loc[alive_ids, 'height'] + adj_h
    out2.loc[alive_ids, 'agb'] = sub.loc[alive_ids, 'agb'] + adj_a
out2.reset_index(inplace=False).to_csv('result/probe_co2_adj_strong.csv', index=False)
print('wrote result/probe_co2_adj_strong.csv (2x adjustment)')
