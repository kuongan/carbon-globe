"""How much does the LSTM-XL prediction move when the val inputs get the TEST format? (no submission needed)

Run in the same Kaggle notebook AFTER lstm_xl_probe.py (it reuses MODEL, rollout, normData, CO2_FEATS, X_MEAN, X_STD,
Y_MEAN, BASE, COMP, SCENARIOS). To skip the test rollouts there, set VARIANTS = [] at the top of lstm_xl_probe.py.

Variants on the 852 val sites (truth = ED run with the historical forcing):
  orig       : as stored                         -> should give VAL 0.0379 (the LSTM-XL run's best epoch)
  clim       : features 36,40,...,128 -> the site's monthly climatology, repeated every year (test format)
  co2_<scn>  : CO2 block 8-31 -> the test scenario's site-mean CO2 series (CO2 is ~uniform in space)
  fmt_<scn>  : clim + co2_<scn> (the full test format we can reproduce)
  co2nos     : CO2 block -> its own annual mean (seasonal cycle removed, historical level and trend kept)
  co2lvl_<scn>: CO2 block -> the test scenario's annual level/trend + the site's historical seasonal cycle
               (co2nos and co2lvl split the co2_<scn> effect into "no seasonal cycle" and "different level")
'msd vs orig' = how far the prediction moves (competition units). Compare it with the ~0.13 common error on the LB.
"""
import gc
import numpy as np
import pandas as pd

CLIM_FEATS = list(range(36, 129, 4))
# run 1 (2026-10-04) covered orig, clim, co2_<scn>, fmt_<scn>; run 2 splits the CO2 effect
VAL_VARIANTS = ['orig', 'co2nos'] + [f'co2lvl_{s}' for s in ('ssp126', 'ssp585')]

def scored_metric(pred, true):
    mh = float(np.mean(((pred[..., 0] - true[..., 0]) / Y_MEAN[0]) ** 2))
    ma = float(np.mean(((pred[..., 1] - true[..., 1]) / Y_MEAN[1]) ** 2))
    return 0.5 * (mh + ma), mh, ma

def msd(a, b):
    return 0.5 * (np.mean(((a[..., 0] - b[..., 0]) / Y_MEAN[0]) ** 2) + np.mean(((a[..., 1] - b[..., 1]) / Y_MEAN[1]) ** 2))

_res = np.load(f'{BASE}/data_global/res_train4_test8.npz')
xv = _res['x_test'].astype(np.float32)                    # (852, 40, 12, 136) raw
yv = _res['y_test'][..., 11, :].astype(np.float32)        # (15, 852, 41, 7) annual states
del _res; gc.collect()
val_y0, val_true = yv[:, :, 0], yv[:, :, 40]

CO2_TEST = {}
for scn in SCENARIOS:
    _tx = np.load(f'{COMP}/test_{scn}.npz')['test_x']
    CO2_TEST[scn] = _tx[..., CO2_FEATS].mean(axis=0)       # (40, 12, 24)
    del _tx; gc.collect()

def variant(name):
    x = xv.copy()
    if name in ('clim',) or name.startswith('fmt_'):
        x[..., CLIM_FEATS] = x[..., CLIM_FEATS].mean(axis=1, keepdims=True)
    if name.startswith(('co2_', 'fmt_')):
        x[..., CO2_FEATS] = CO2_TEST[name.split('_')[1]][None]
    if name == 'co2nos':
        x[..., CO2_FEATS] = x[..., CO2_FEATS].mean(axis=2, keepdims=True)
    if name.startswith('co2lvl_'):
        c = x[..., CO2_FEATS]
        seas = c - c.mean(axis=2, keepdims=True)                                      # historical seasonal cycle
        x[..., CO2_FEATS] = CO2_TEST[name.split('_')[1]].mean(axis=1, keepdims=True)[None] + seas
    return x

rows, ref = [], None
for name in VAL_VARIANTS:
    x = variant(name)
    pred = rollout(lambda t: normData(x[:, t], X_MEAN, X_STD).astype(np.float32), x.shape[0], val_y0, desc=name)
    sc, mh, ma = scored_metric(pred, val_true)
    ref = pred if name == 'orig' else ref
    rows.append(dict(variant=name, val_metric=sc, mse_h=mh, mse_agb=ma, msd_vs_orig=msd(pred, ref),
                     bias_h=float(np.mean(pred[..., 0] - val_true[..., 0]) / Y_MEAN[0]),
                     bias_agb=float(np.mean(pred[..., 1] - val_true[..., 1]) / Y_MEAN[1]),
                     agb_zero=float(np.mean(pred[..., 1] <= 1e-6))))
    del x, pred; gc.collect()
print(pd.DataFrame(rows).round(5).to_string(index=False))
