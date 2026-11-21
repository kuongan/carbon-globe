# CarbonGlobe AI Emulation Challenge: team iuem (14th / 288)

Our solution to the **IEEE BigData Cup 2026 AI Emulation Challenge** on Kaggle. The task is to emulate the ED forest
simulator: predict forest height and biomass after 40 years of future climate.

| | |
|---|---|
| Private leaderboard | **14th of 288** (0.046) |
| Final submission | `final/submission_final.csv`: public 0.046, private 0.047 |
| Provided baseline | 0.231 |
| Team | An Nguyen Tran Khuong, Phu Nguyen Thi Minh (University of Information Technology, VNU-HCM) |

**Start here**

- [WRITEUP.md](WRITEUP.md): the full story. Data findings, models, leaderboard-based blending, the public/private
  shake-up, and what failed.
- [final/iuem_report.pdf](final/iuem_report.pdf): the 2-page technical report submitted to the organizers.
- [final/README.md](final/README.md): exact reproduction of the final submission, model by model.

**Reproduce the final file** (CPU, a few minutes, from the saved model outputs in `result/`):

```bash
pip install -r requirements.txt lightgbm==4.6.0 scipy
python final/make_final.py
```

## Repository map

| Path | Content |
|---|---|
| `final/` | Final blend (`make_final.py`), report, reproduction guide, submitted CSV |
| `carbonglobe_direct_nn.py` | Cross-age neural model (best single model) |
| `carbonglobe_gbm.py` | LightGBM direct / random-forest / rollout models, chronosequence and event-risk features |
| `carbonglobe-global.ipynb`, `global_probe.py` | LSTM-XL grid emulator: training and test-time inference |
| `carbonglobe-ecoformer.ipynb` | EcoFormer emulator |
| `analog_build.py`, `analog.py` | Own-site analog member |
| `test_transform.py`, `dead_rows.py`, `slope.py` | Recovering the test-space affine maps, dead stands, mean matching |
| `ensemble.py`, `probe_bias.py`, `cv_parity.py` | Blending and score prediction from leaderboard scores |
| `chrono.py`, `fireprobe.py`, `eda*.py`, `*probe.py` | Data analysis behind the findings in the write-up |
| `kaggle_notebooks/`, `make_notebooks.py` | Ready-to-run Kaggle notebooks (GPU / large-RAM steps) |
| `result/` | Saved test predictions of the models (inputs to the blend) |
| `INSIGHTS.md`, `PLAN.md` | Lab notebook: every submission and hypothesis (Vietnamese) |
| `LSTM-LSTNet-DeepED/`, `Crossformer/`, `DLinear/`, `Informer-Transformer/`, `TimeXer/` | Baseline models of the upstream CarbonGlobe repository |

Data are not included. The competition files and the
[CarbonGlobe dataset](https://www.kaggle.com/datasets/zhihaow/carbonglobe) go under `data/`, as described in
`final/README.md`. Trained weights are not in git because of size.

## Credits

This repository is a fork of the CarbonGlobe benchmark repository (Wang et al., NeurIPS 2025 Datasets & Benchmarks).
Its original README is [CARBONGLOBE.md](CARBONGLOBE.md), and its license applies ([LICENSE.txt](LICENSE.txt)).
