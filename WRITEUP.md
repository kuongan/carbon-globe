# IEEE BigData Cup 2026 AI Emulation Challenge: team iuem solution (14th / 288)

**Team iuem**: An Nguyen Tran Khuong, Phu Nguyen Thi Minh (University of Information Technology, VNU-HCM).

**Result**: 14th of 288 teams on the private leaderboard (0.046). Our identified final submission
`submission_final.csv` scored 0.046 public / 0.047 private. The provided DeepED baseline scored 0.231, and
predicting "no change from year 0" scored 0.285. The metric is MSE, so lower is better.

The technical report is [final/iuem_report.pdf](final/iuem_report.pdf) (2 pages). This write-up is the longer
story: what we found in the data, what we built, what failed, and what the private leaderboard taught us.

## 1. The task

The task is to emulate ED (Ecosystem Demography), a process-based forest simulator. For each stand, you get its
state at year 0 and 40 years of future monthly climate, and you predict its state at year 40.

- **Training data**: the CarbonGlobe grid, with 54,152 sites at 0.5°. It covers historical climate (136 inputs per
  month), stands started at 15 ages (1 to 500 years), and 7 state variables per year.
- **Test data**: 6,171 sites under two future scenarios, SSP1-2.6 and SSP5-8.5.
- **Scoring**: only height and above-ground biomass (agb) at year 40 are scored, as ½(MSE<sub>height</sub> +
  MSE<sub>agb</sub>) on values divided by fixed constants.

## 2. What the data told us

Most of our progress came from understanding the data rather than from bigger models.

**The test is the training grid, stored in different units.** Every test variable is an affine transform of the
same quantity on the grid: *y*<sub>test</sub> = *s*·*y* + *b*, with one slope and one offset per variable. After
inverting these maps on the 7 static site features, 97% of the test sites coincide with grid sites. We recover the
maps in `test_transform.py`.

- Height is exact: *s* = 0.876, *b* = −0.901. ED's 35 m height cap lands exactly on the test maximum.
- The other slopes are fitted by regression. For agb, *s* ≈ 0.84.
- The hidden answers use the same transform. We confirmed this with a probe: setting dead stands to the mapped zero
  scored 0.072, and setting them to physical zero would have scored 0.074 (`probe_dead.py`).

**Dead stands are known.** 8.8% of test rows start at *b*, the transformed value of 0. On the grid, such stands
stay at 0 for all 40 years without exception, so their answer is exactly *b*.

**Leave the published climate alone.** We tried mapping the test climate back to physical units, correcting it,
and adding CO2 in every form. All of it made models worse.

**A site's own ages reveal its growth curve.** The 15 seed ages of one site share the same climate. Their year-0
states therefore trace ED's growth curve at that site, which we call a *chronosequence*. The year-0 state of the
stand that is 40 years older is already a strong guess for year 40. For age 1 its error is 0.037, against 0.93 for
persistence (`chrono.py`).

**Collapses are climate events.** 22% of old stands lose more than 30% of agb in a single year, and every age of
the site collapses in the same year (fire or drought in ED). That year's monthly climate predicts the event with
AUC 0.985 (`fireprobe.py`).

Adding chronosequence and event-risk features took a LightGBM direct model from 0.097 to 0.073 on the leaderboard,
and from 0.0305 to 0.0245 on the grid holdout.

## 3. Models

All models are trained on the grid in physical units, without the CO2 block. The test inputs are converted before
inference, and the predictions are converted back to test units afterwards.

| Model | Idea | Script |
|---|---|---|
| Cross-age neural model | Predicts all 15 ages of a site at once. Conv1d + BiGRU encode the 40 years of climate; one token per age carries the encoding, year-0 state, chronosequence and event risk; a 2-layer Transformer attends across ages. | `carbonglobe_direct_nn.py` |
| LightGBM | Direct year-40 model (with and without the engineered features), random-forest mode, and an annual rollout. | `carbonglobe_gbm.py` |
| LSTM-XL emulator | Steps from one year to the next and is rolled out 40 times. Based on the baseline repo. | `carbonglobe-global.ipynb`, `global_probe.py` |
| EcoFormer | Transformer encoder–decoder emulator. Based on the baseline repo. | `carbonglobe-ecoformer.ipynb` |
| Own-site analog | Test year-0 state plus the historical 40-year change of the matched grid site. No training. | `analog_build.py`, `analog.py` |

Two details mattered a lot:

**How to feed the test to a model trained in physical units.** We tried four input versions:

- `none`: inputs as published;
- `y0inv`: year-0 state mapped back to physical units;
- `xinv`: climate mapped back;
- `y0inv2` = 2·y0<sub>phys</sub> − y0<sub>test</sub>: one more step in the same direction.

`y0inv2` won for every model family. We have no physical explanation for it, so treat it as an empirical
heuristic.

**How to map predictions back.** `map` applies *s*·*p* + *b*. `match` keeps *b* exact and picks the slope so that
living stands get the mean of an earlier scored submission. This calibrates the agb slope, which we could only
estimate.

## 4. Blending with the leaderboard as a measuring device

For MSE, the errors of any two models *i* and *j* satisfy

  2·Cov(*e<sub>i</sub>*, *e<sub>j</sub>*) = MSE<sub>i</sub> + MSE<sub>j</sub> − D<sub>ij</sub>,

where D<sub>ij</sub> is the mean squared difference of their two prediction files. D<sub>ij</sub> is computable
without the truth. So each submission's public score gives the full error covariance **C** of every model we ever
scored, and the best weights summing to 1 are **w** = **C**<sup>−1</sup>**1** / **1**<sup>T</sup>**C**<sup>−1</sup>**1**.

We exclude the dead rows, whose answer is known. Using this, we predicted the score of every blend before spending
a submission (`ensemble.py`, `probe_bias.py`).

- For blends of up to 8 members, the predictions matched the leaderboard within ±0.001.
- With 12–16 members, the predictions were 0.002–0.003 too optimistic.

The final blend has 12 of the 63 models we scored, chosen by greedy forward selection with swaps. The weights are
in [final/README.md](final/README.md) and the report's Table 2. Several weights are negative, down to −0.335. They
extrapolate along pairs of closely related models, for example the same LSTM fed two input versions.

## 5. Public vs private: what the shake-up says

| Submission | Type | Public | Private |
|---|---|---|---|
| Cross-age neural model (best single) | single model | 0.061 | **0.056** |
| LightGBM + chronosequence + event risk | single model | 0.073 | 0.072 |
| x7f | two-level blend (a blend + 2 models) | 0.054 | 0.054 |
| x9f | two-level blend (a blend + 6 models) | 0.049 | **0.047** |
| flat6 | 6 single models | 0.049 | 0.048 |
| flatg12 = `submission_final.csv` (final) | 12 single models | **0.046** | 0.047 |
| flatg16 | 16 single models | 0.046 | 0.046 |

- **Single models improved on private** (the NN went from 0.061 to 0.056). Their public scores were not fitted to
  anything.
- **Big blends gave some of it back.** The 12-member final lost 0.001, while the smaller x9f gained 0.002. Weights
  solved from public scores, with large negative weights, fit the public split's noise. Our own optimism
  measurement (0.002–0.003 for 12–16 members) predicted exactly this.
- **Lesson.** Beyond about 8 members, the covariance blend stops paying for its variance. A smaller blend with
  fewer negative weights would have been the safer final.

## 6. What did not work

- Correcting the test climate, or using CO2 as an input in any form.
- Per-site corrections of the blend (CO2, climate, growth, height-vs-agb).
- Per-scenario or per-age blend weights: +0.0005, inside the rounding noise.
- Correcting errors per site from history: errors are not persistent across periods.
- More LSTM snapshots, bagging, fine-tuning and larger LSTMs.
- Per-year event probabilities, event magnitude, monthly climate in the NN, climate fingerprints, a
  chronosequence-anchored target, y0 augmentation, and an agb slope of 0.876. All of these gave an equal or better
  grid holdout but no leaderboard gain.
- **Trusting the grid holdout.** It ranks models within a family, but not against the leaderboard: the test climate
  is future, the holdout's is historical. Every final decision used the leaderboard algebra above.

The full lab notebook is [INSIGHTS.md](INSIGHTS.md) (Vietnamese). It covers every submission (#1–#170) and every
verified or refuted hypothesis.

## 7. Reproduce

Everything runs from the repository root. Details per model are in [final/README.md](final/README.md).

```bash
pip install -r requirements.txt
python final/make_final.py          # rebuilds final/submission_final.csv from the saved model outputs (minutes, CPU)
```

The saved test predictions of all 12 members are in `result/`. Regenerating them needs the trained weights, which
are kept out of git because of size: LightGBM boosters, NN checkpoints, LSTM and EcoFormer weights. With the
weights, every model has an inference-only command or Kaggle notebook in `final/README.md` and `kaggle_notebooks/`.
Training from scratch uses the same scripts without the weights flags; GPU work was done on Kaggle T4 sessions.

Data: the competition files and the CarbonGlobe dataset ([Kaggle](https://www.kaggle.com/datasets/zhihaow/carbonglobe),
[Hugging Face](https://huggingface.co/datasets/zhwang1/CarbonGlobe)) go under `data/` as described in
`final/README.md`. They are not part of this repository.

## Credits

- This repository is a fork of the CarbonGlobe benchmark repository (Zhihao Wang et al., NeurIPS 2025 Datasets &
  Benchmarks). Its README is kept as [CARBONGLOBE.md](CARBONGLOBE.md).
- The LSTM-XL and EcoFormer emulators build on `LSTM-LSTNet-DeepED/` from that repository.
- ED: Moorcroft, Hurtt & Pacala (2001). LightGBM: Ke et al. (2017).
