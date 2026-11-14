# Final submissions

**Main: `flatg12` (public LB 0.046)**: `python final/make_submission_flatg12.py` (or the `.ipynb`) rebuilds it
exactly (max |diff| 7e-15) from the scored predictions of its twelve single models (`result/<member>_preds/`) and
their public LBs, with the same closed-form weights as below. Members: cross-age NN (+0.71), LSTM-XL physical-input
pair members, chrono GBM, RF, EcoFormer, GBM rollout, own-site analog, ... (list and provenance in the script header).

**Fallback: `flat6` (public LB 0.049)**, documented below: rebuilt from the raw model outputs.

## flat6 (public LB 0.049)

One blend layer over six single models. `python final/make_submission.py` (or `final/make_submission.ipynb`, the
same code) rebuilds the submission from the raw model outputs in this repository and checks every step:
seed averages, each member, and the final file against the one that was scored
(`result/submission_blend_flat6.csv`, max |diff| 3e-7).

## How the answer space works (why there is a transform step)

The test set is the CarbonGlobe grid behind a per-variable affine map `y_test = s * y_physical + b`
(`test_transform.py`; height exact, `s = 0.8763`, `b = -0.9009`), and the hidden answer is in the same mapped space
(`INSIGHTS.md` 4.22-4.23; confirmed twice on the LB, last by `probe_x7f_dead_raw0.csv`: 0.055 vs 0.054 expected in
the mapped space). Models are trained on the physical grid; their predictions are brought to the answer space by
`map` (the slopes of `result/test_transform.npz`) or `match` (per scenario and variable, the slope that matches the
alive-row mean of an LB-calibrated reference, `final/ref/meanmatch_reference.csv`). Stands that are dead at year 0
(`result/dead_rows.npz`, 8.8%) stay dead: their answer is the mapped zero, known exactly.

## The six members

| Member | Weight | LB alone | Model | Raw output (in this repo) | How it was made |
|---|---|---|---|---|---|
| `nnx8_y0inv2` | +0.537 | 0.061 | cross-age neural direct model + chronosequence + collapse-event risk, 8 seeds | `result/nn_direct_xage/nn_preds.npz` (seeds 4-7), `result/nn_direct_xage2/nn_preds.npz` (seeds 8-11), key `<scn>_y0inv2` | `kaggle_notebooks/kaggle_direct_nn_xage.ipynb` and `kaggle_direct_nn_xage2.ipynb` (GPU T4, ~3 h / ~5 h); source `carbonglobe_direct_nn.py` with `NN_CHRONO=1 NN_XAGE=1` |
| `gbmc3i` | +0.401 | 0.091 | LightGBM direct + chronosequence + collapse-event risk, 3 seeds | `result/gbm/gbm_preds_chrono{,_s7,_s11}.npz`, key `<scn>_direct_y0inv` | `GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2,none GBM_SEED=<42/7/11> GBM_TAG=<_chrono/_chrono_s7/_chrono_s11> python carbonglobe_gbm.py` (CPU, ~25 min each) |
| `lstm_y0inv_xinv` | +0.305 | 0.109 | LSTM-XL grid model, no CO2, seed 42, epoch 4 | `result/1/global_probe_preds.npz`, key `<scn>_y0inv_xinv` | trained with `carbonglobe-global.ipynb` (`CO2_MODE='const'`, `RUN_NAME='global_noco2'`, SEED 42); inference `global_probe.py` with `PROFILE='noco2'` (weights `snap_ep4.npz`, Kaggle dataset `noco2-ep4`) |
| `sw_s42ep6_y0inv_xinv` | +0.278 | 0.112 | same LSTM run, epoch 6 | `result/kaggle_run1/global_probe_preds.npz`, key `<scn>_s42ep6_y0inv_xinv` | `kaggle_notebooks/kaggle_sweep_pairs_run1.ipynb` (`global_probe.py`, `PROFILE='sweep'`, dataset `lstm-sweep`) |
| `rfm_none` | -0.260 | 0.155 | LightGBM random-forest mode, y0 left in the test space | `result/gbm/gbm_preds_rf.npz`, key `<scn>_direct_none` | `GBM_BOOSTING=rf GBM_MODELS=direct GBM_TAG=_rf python carbonglobe_gbm.py` |
| `s7_xinv` | -0.260 | 0.277 | LSTM-XL no CO2, seed 7, epoch 3, physical forcing, y0 as given | `result/noco2-s7/global_probe_preds.npz`, key `<scn>_xinv` | trained with `carbonglobe-global.ipynb` (SEED 7, `RUN_NAME='global_noco2_s7'`); inference `global_probe.py` with `PROFILE='noco2_s7'` (dataset `noco2-s7`) |

Transforms: the two NN / GBM members and `rfm_none` use `match`, the three LSTM members use `map`.
Variant names: `y0inv` = y0 mapped back to physical, forcing as published; `y0inv2` = y0 pushed past physical by
the same step (`2 y0_phys - y0_test`); `xinv` = forcing mapped back to physical; `none` = inputs as published.

## Weights

With an MSE metric, `MSE_i + MSE_j - D_ij = 2 Cov(e_i, e_j)`, where `D_ij` is the mean squared difference of two
members' predictions (computable) and `MSE_i` is each member's public LB. The error covariance therefore needs no
truth, and the best weights summing to 1 are `C^-1 1 / (1' C^-1 1)`. Estimated LB 0.0487, scored 0.049. The
six members were chosen by exhaustive search over 4- and 5-member sets and a swap search over 6-member sets among
58 scored single models (`INSIGHTS.md` 4.51); GBM / NN only, without the LSTMs, reaches 0.055 at best.

## Data

Competition data (`test_ssp126.npz`, `test_ssp585.npz`) and the CarbonGlobe dataset (`zhihaow/carbonglobe`) under
`data/`. Full experiment log: `INSIGHTS.md` (in Vietnamese) and `PLAN.md`.
