# IEEE BigData Cup 2026 AI Emulation Challenge: team **iuem**: reproduction guide

Final Kaggle submission: **`submission_final.csv`**, produced by `final/make_final.py` (a fixed-weight blend of eight
single models). Method: `final/REPORT.md`. Everything below runs from the repository root.

## 1. Environment

| | |
|---|---|
| Python | 3.10+ (developed on 3.11 / Kaggle image) |
| Packages | `numpy` (>= 1.26), `pandas` (>= 2.0), `lightgbm==4.6.0`, `torch` (>= 2.1, CUDA for training), `tensorflow` 2.x (LSTM inference; the Kaggle GPU image works as is), `scikit-learn` |
| Hardware | GPU with >= 16 GB (Kaggle T4) for the neural models; CPU with ~30 GB RAM for LightGBM (Kaggle CPU session) |
| Time, inference only from the supplied weights | LSTM ~1 h per profile (GPU), NN ~20 min (GPU), GBM ~1-2 h (CPU, mostly data loading), blend < 2 min |

## 2. Data organization

```
data/public/test_ssp126.npz, test_ssp585.npz            # competition data (Kaggle: /kaggle/input/competitions/.../public)
data/carbonglobe/data_global/glob_X_fea.npy, glob_Y*.npy  # CarbonGlobe dataset (Kaggle: zhihaow/carbonglobe)
data/carbonglobe/data_stats/data_stats.npz, mask2d.npy
kaggle_upload/noco2-ep4/test_maps.npz                     # test-space affine maps (supplied; written by test_transform.py)
result/dead_rows.npz, result/test_transform.npz           # supplied (in the repository)
final/ref/meanmatch_reference.csv                         # mean-matching reference (supplied)
```
On Kaggle the scripts find the inputs under `/kaggle/input` themselves (attach the competition data, `zhihaow/carbonglobe`
and a dataset holding `test_maps.npz`). `global_probe.py` (LSTM inference) reads `/kaggle/input` only: run it as a Kaggle
notebook, or change `ROOT_IN` at its top.

## 3. Supplied weights (folder `weights/`)

| Member (blend weight) | Model | Files |
|---|---|---|
| `nnx_y0inv2` (+0.651) | cross-age neural direct model, 4 seeds | `weights/nn/nn_seed{4,5,6,7}.pt`, `nn_event_clf.txt`, `nn_prep.npz` |
| `gbmc3i` (+0.343) | LightGBM direct + chronosequence + event risk, seeds 42 / 7 / 11 | `weights/gbm/gbm_direct_chrono{,_s7,_s11}_{0,1}.txt`, `gbm_event_chrono{,_s7,_s11}.txt` |
| `lstm_y0inv_xinv` (+0.344) | LSTM-XL grid model, no CO2, seed 42, epoch 4 | `weights/lstm/snap_ep4.npz` |
| `sw_s42ep6_y0inv_xinv` (+0.310) | same run, epoch 6 | `weights/lstm/sweep_s42ep6.npz` |
| `s7_xinv` (-0.300) | LSTM-XL no CO2, seed 7, epoch 3 | `weights/lstm/s7_ep3.npz` |
| `rfm_none` (-0.228) | LightGBM random-forest mode | `weights/gbm/gbm_direct_rf_{0,1}.txt` |
| `gbmm_y0inv2` (-0.206) | LightGBM direct without chronosequence features | `weights/gbm/gbm_direct_y0x_{0,1}.txt` |
| `gbm_rollout_y0inv` (+0.086) | LightGBM annual rollout (7 state models) | `weights/gbm/gbm_rollout_roll_{0..6}.txt` |

## 4. Reproduce the final predictions from the weights (no training)

**Step 1: LSTM members** (`global_probe.py`, TensorFlow, GPU; one Kaggle notebook per profile, the weights file attached
as a dataset next to `test_maps.npz`). Each run writes `global_probe_preds.npz` (physical predictions):
```
PROBE_PROFILE=noco2     python global_probe.py   # weights snap_ep4.npz   -> key <scn>_y0inv_xinv   (lstm_y0inv_xinv)
PROBE_PROFILE=sweep     python global_probe.py   # weights sweep_s42ep6.npz (set SWEEP_ONLY = ('s42ep6',)) -> <scn>_s42ep6_y0inv_xinv
PROBE_PROFILE=noco2_s7  python global_probe.py   # weights s7_ep3.npz     -> key <scn>_xinv          (s7_xinv)
```
The repository already holds these three outputs (`result/1/`, `result/kaggle_run1/`, `result/noco2-s7/`
`global_probe_preds.npz`); each run checks itself against the scored predictions (`reproduction ...` log line).

**Step 2: neural member** (PyTorch; GPU recommended):
```
NN_CHRONO=1 NN_XAGE=1 NN_SEEDS=4,5,6,7 NN_WEIGHTS=weights/nn python carbonglobe_direct_nn.py   # -> nn_preds.npz
```

**Step 3: LightGBM members** (CPU; one run per configuration, same environment variables as in training):
```
W=weights/gbm
GBM_WEIGHTS=$W GBM_TAG=_chrono     GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2 GBM_SEED=42 python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_chrono_s7  GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2 GBM_SEED=7  python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_chrono_s11 GBM_CHRONO=1 GBM_MODELS=direct GBM_VARIANTS=y0inv,y0inv2 GBM_SEED=11 python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_rf   GBM_BOOSTING=rf GBM_MODELS=direct GBM_VARIANTS=y0inv,none python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_y0x  GBM_MODELS=direct  GBM_VARIANTS=y0inv,y0inv2 python carbonglobe_gbm.py
GBM_WEIGHTS=$W GBM_TAG=_roll GBM_MODELS=rollout GBM_VARIANTS=y0inv        python carbonglobe_gbm.py
```
Outputs go to `result/gbm/` locally (`/kaggle/working` on Kaggle): `gbm_preds_<tag>.npz`.

**Step 4: blend**:
```
python final/make_final.py --nn <nn_preds.npz> --gbm <dir with gbm_preds_*.npz> \
    --lstm-ep4 <...> --lstm-sweep <...> --lstm-s7 <...> --check submission_final.csv
```
writes `final/submission_final.csv` and prints its max difference to the submitted file.

## 5. Training from scratch (optional)

| Member | Notebook (Kaggle) / command | Hardware, time |
|---|---|---|
| NN | `kaggle_notebooks/kaggle_direct_nn_xage.ipynb` (= `carbonglobe_direct_nn.py`, `NN_CHRONO=1 NN_XAGE=1`) | GPU T4, ~3 h |
| GBM (all 5) | `kaggle_notebooks/kaggle_final_gbm.ipynb` (= `carbonglobe_gbm.py`, six configurations above, no `GBM_WEIGHTS`) | CPU, ~5-7 h |
| LSTM | `carbonglobe-global.ipynb` (`CO2_MODE='const'`, SEED 42 / 7) | GPU T4 x2, ~5-6 h per run |

Training writes the weights listed in section 3 next to the predictions. The blend weights in `final/make_final.py`
were solved once from the public LB of each member and are kept fixed.
