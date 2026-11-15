# IEEE BigData Cup 2026 AI Emulation Challenge: team **iuem**: reproduction guide

Final Kaggle submission: **`submission_final.csv`** (= `final/submission_final.csv`), produced by
`python final/make_final.py`: a fixed-weight blend of **twelve** single models (`flatg12`). Method: `final/REPORT.md`.
Everything below runs from the root of the code package (`iuem_code.zip`, no GitHub access needed: all code,
including `LSTM-LSTNet-DeepED/` for EcoFormer, and every raw output used by the blend are inside it; on Kaggle,
attach the extracted package as a dataset). Technical report: `final/iuem_report.pdf`.

## 1. Environment

| | |
|---|---|
| Python | 3.10+ (developed on 3.11 and the Kaggle image) |
| Packages | `numpy` (>= 1.26), `pandas` (>= 2.0), `scipy`, `lightgbm==4.6.0`, `torch` (>= 2.1), `tensorflow` 2.x (LSTM / EcoFormer inference; the Kaggle GPU image works as is), `scikit-learn` |
| Hardware | GPU with >= 16 GB (Kaggle T4) for the neural models; CPU with ~30 GB RAM for LightGBM |
| Time, inference only | LSTM ~30 min per weight set (GPU), NN ~20 min (GPU), EcoFormer ~30 min (GPU), GBM ~1-2 h per configuration (CPU, mostly data loading), analog + blend a few minutes (CPU) |

## 2. Data organization

```
data/public/test_ssp126.npz, test_ssp585.npz            # competition data (Kaggle: /kaggle/input/competitions/.../public)
data/carbonglobe/data_global/glob_X_fea.npy, glob_Y*.npy  # CarbonGlobe dataset (Kaggle: zhihaow/carbonglobe)
data/carbonglobe/data_stats/data_stats.npz, mask2d.npy
kaggle_upload/noco2-ep4/test_maps.npz                     # test-space affine maps (= weights/test_maps.npz)
result/dead_rows.npz, result/test_transform.npz           # in the repository
final/ref/meanmatch_reference.csv                         # mean-matching reference (in the repository)
```
On Kaggle the scripts find their inputs under `/kaggle/input` (attach the competition data, `zhihaow/carbonglobe` and a
dataset holding the weights + `test_maps.npz`). `global_probe.py` (LSTM inference) reads `/kaggle/input` only: run it
as a Kaggle notebook, or change `ROOT_IN` at its top.

## 3. The twelve members and their weights (folder `weights/`)

| Member | Blend weight | Model | Weights | Raw output used by `make_final.py` |
|---|---|---|---|---|
| `nnx_y0inv2` | +0.707 | cross-age neural direct model (chronosequence + event risk), 4 seeds | `nn/nn_seed{4..7}.pt`, `nn/nn_event_clf.txt`, `nn/nn_prep.npz` | `result/final_runs/nn/nn_preds.npz` |
| `sw_s42ep6_y0inv_xinv` | +0.373 | LSTM-XL, no CO2, seed 42, epoch 6; physical y0 + forcing | `lstm/sweep_s42ep6.npz` | `result/kaggle_run1/global_probe_preds.npz` |
| `lstm_y0inv_xinv` | +0.349 | LSTM-XL, no CO2, seed 42, epoch 4; physical y0 + forcing | `lstm/snap_ep4.npz` | `result/1/global_probe_preds.npz` |
| `gbmc3i` | +0.337 | LightGBM direct + chronosequence + event risk, seeds 42 / 7 / 11; physical y0 | `gbm/gbm_direct_chrono{,_s7,_s11}_{0,1}.txt`, `gbm/gbm_event_chrono{,_s7,_s11}.txt` | `result/final_runs/gbm/gbm_preds_chrono{,_s7,_s11}.npz` |
| `s7_ep3` | +0.171 | LSTM-XL, no CO2, seed 7, epoch 3; inputs as published | `lstm/s7_ep3.npz` | `result/noco2-s7/global_probe_preds.npz` (`none`) |
| `ecoformer_ed` | +0.129 | EcoFormer encoder-decoder | `ecoformer/ecoformer_ed.weights.h5` | `result/ecoformer_run/ecoformer_ed_preds.npz` |
| `gbm_rollout_y0inv` | +0.106 | LightGBM annual rollout (7 state models) | `gbm/gbm_rollout_roll_{0..6}.txt` | `result/final_runs/gbm/gbm_preds_roll.npz` |
| `analog_add` | -0.130 | own-site historical growth added to the test y0 (no training) | none (code: `analog_build.py`, `analog.py`) | `result/analog_preds.npz` |
| `rfm_none` | -0.199 | LightGBM random-forest mode; y0 as published | `gbm/gbm_direct_rf_{0,1}.txt` | `result/final_runs/gbm/gbm_preds_rf.npz` |
| `gbmm_y0inv2` | -0.199 | LightGBM direct without chronosequence features; y0 past physical | `gbm/gbm_direct_y0x_{0,1}.txt` | `result/final_runs/gbm/gbm_preds_y0x.npz` |
| `nov2_snapmean` | -0.309 | LSTM-XL run 5 (no CO2, group v2 at its mean), mean of 5 weight sets | `lstm/nov2/nov2_{best,ep2,ep3,ep4,ep5}.npz` | `result/final_runs/nov2/global_probe_preds.npz` |
| `s7_xinv` | -0.335 | LSTM-XL seed 7 epoch 3; physical forcing | `lstm/s7_ep3.npz` | `result/noco2-s7/global_probe_preds.npz` (`xinv`) |

Weights sum to 1; negative weights extrapolate along pairs of variants of one model (closed-form blend, REPORT §3).
All raw outputs listed are in the repository, so step 5 alone rebuilds the submission in minutes.

## 4. Reproduce the raw outputs from the weights (no training)

**LSTM members** (`global_probe.py`, TensorFlow, GPU; one Kaggle notebook per profile, weights attached as a dataset):
```
PROBE_PROFILE=noco2    python global_probe.py   # lstm/snap_ep4.npz                -> <scn>_y0inv_xinv
PROBE_PROFILE=sweep    python global_probe.py   # lstm/sweep_s42ep6.npz (SWEEP_ONLY = ('s42ep6',)) -> <scn>_s42ep6_y0inv_xinv
PROBE_PROFILE=noco2_s7 python global_probe.py   # lstm/s7_ep3.npz                  -> <scn>_none, <scn>_xinv
PROBE_PROFILE=nov2     python global_probe.py   # lstm/nov2/nov2_*.npz             -> <scn>_snapmean_none
```
Ready notebook for `nov2`: `kaggle_notebooks/kaggle_nov2_infer.ipynb` (checked: identical to the training run).

**EcoFormer**: `kaggle_notebooks/kaggle_ecoformer_infer.ipynb` (`MAX_EPOCHS = 0`, the weights loaded; the last cell
compares with the scored predictions).

**Neural member** (PyTorch, GPU recommended; data loading needs ~25 GB RAM):
```
NN_CHRONO=1 NN_XAGE=1 NN_SEEDS=4,5,6,7 NN_WEIGHTS=weights/nn python carbonglobe_direct_nn.py   # -> nn_preds.npz
```

**LightGBM members** (CPU; same environment variables as in training, plus `GBM_WEIGHTS`):
```
W=weights/gbm
GBM_WEIGHTS=$W GBM_TAG=_chrono     GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2 GBM_SEED=42 python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_chrono_s7  GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2 GBM_SEED=7  python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_chrono_s11 GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2 GBM_SEED=11 python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_rf   GBM_BOOSTING=rf GBM_MODELS=direct GBM_VARIANTS=y0inv,none python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_y0x  GBM_MODELS=direct  GBM_VARIANTS=y0inv,y0inv2 python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_roll GBM_MODELS=rollout GBM_VARIANTS=y0inv        python carbonglobe_gbm.py
```
**Own-site analog** (CPU, no weights): `python analog_build.py && python analog.py final/ref/meanmatch_reference.csv`.

## 5. Blend

```
python final/make_final.py --check submission_final.csv          # defaults = the raw outputs listed in section 3
```
writes `final/submission_final.csv`; pass `--nn / --gbm / --nov2 / --ecoformer / --analog / --lstm-*` to use freshly
regenerated raw outputs instead. `--blend repro8` builds the 8-member fallback.

## 6. Training from scratch (optional)

| Member | Notebook / command | Hardware, time |
|---|---|---|
| NN | `kaggle_notebooks/kaggle_direct_nn_xage.ipynb` (`carbonglobe_direct_nn.py`, `NN_CHRONO=1 NN_XAGE=1`, seeds 4-7) | GPU T4, ~3 h |
| 5 LightGBM members | `kaggle_notebooks/kaggle_final_gbm.ipynb` (the six configurations above without `GBM_WEIGHTS`) | CPU, ~5-7 h |
| LSTM-XL runs | `carbonglobe-global.ipynb` (`CO2_MODE='const'`; SEED 42; SEED 7; `RUN_NAME='global_nov2'`, `DROP_GROUPS=('v2',)`) | GPU T4 x2, ~5-6 h per run |
| EcoFormer | `carbonglobe-ecoformer.ipynb` | GPU T4, ~3 h |

The blend weights in `final/make_final.py` were solved once from the public LB of each member and are kept fixed.
