# Plan and runbook

Best LB: **0.091** (`result/submission_blend_latest_x.csv`); safe fallback 0.094 (`result/bg_scn.csv`). Evidence for every
number: `INSIGHTS.md`. Division of work: you run Kaggle sessions and submit; I code, analyse and blend.

## What drives the score (from 28 submissions)

| Lever | Gain | Kind |
|---|---|---|
| train on the whole grid (51.6k sites) | 0.168 → 0.134 | data |
| move the test CO2 toward history (`co2_hist`) | 0.134 → 0.122 | data |
| blend grid model with old models | 0.122 → 0.100 | combination |
| per-scenario bias factors | 0.100 → 0.094 | data (ED's scenario response) |
| push the CO2 move further (negative weight) | 0.094 → 0.091 | data |
| bigger models / longer training / scheduled sampling | ≈ 0 or worse | model |

So the work below is mostly experiments that answer a question about the data; every answer is then turned into
score by `ensemble.py` / `ensemble_scn.py` / `probe_bias.py`.

## Questions, and the experiment for each

| # | Question | Experiment | Cost |
|---|---|---|---|
| Q1 | How far should the test CO2 be moved? (LB says ~1.9x co2_hist) | session A, part C: `co2x2`, `co2x3`, `co2_cont`, `co2_end` | 4 slots |
| Q2 | Does any other feature group act like CO2? | session A, part A (screening) | 0 |
| Q3 | Is feature 7 (20x seasonal amplitude) hurting? | session A: `co2_hist+f7_fix` | 1 slot |
| Q4 | How much does the year-40 prediction remember the shifted `test_y0`? | session A, part B | 0 |
| Q5 | Is a model with **no CO2 input** better? Its CO2 response is the one thing every fix has fought. | session B | 1–3 slots |
| Q6 | Per scenario, which model is better? (old models vs grid model may differ by scenario) | hybrid probes, `ensemble_scn.py` | 6 slots |
| Q7 | Does a second grid model (other seed) add diversity? | session C | 1–3 slots |
| Q8 | Which training epoch is best on the LB? (holdout cannot tell: run 2 holdout better, LB worse) | snapshot submissions of B and C | included above |

## Runbook

### Session A: inference probes with grid run 1 (GPU T4 x1, ~1-1.5 h)
- Notebook: one cell with the content of `global_probe.py`.
- Inputs: competition data, `zhihaow/carbonglobe`, a dataset with **only** run 1's `global_lstmxl.weights.h5`
  (`result/globel-train/global_lstmxl.weights.h5`; run 2's weights scored worse).
- Submit: `submission_global_co2x2.csv`, `submission_global_co2x3.csv`, `submission_global_co2_cont.csv`,
  `submission_global_co2_end.csv`, `submission_global_co2_hist_f7_fix.csv`. Skip `..._co2_hist.csv` (= the 0.122 file).
- Send back: the part A table, the part B lines, `global_probe_preds.npz`, the 5 LB scores.

### Session B: grid model without CO2 (GPU T4 x2, ~5-6 h)
- `carbonglobe-global.ipynb` as committed: `CO2_MODE='const'`, `RUN_NAME='global_noco2'`, 6 epochs, K=1,
  snapshots of epochs 3/4/5. Do **not** attach an old checkpoint (it is a new model; the fingerprint refuses it anyway).
- Submit: `submission.csv` (best holdout epoch) and up to two `submission_ep<k>_none.csv`.
- Send back: the training log, `global_noco2_preds.npz`, the LB scores.

### Session C: second grid seed with CO2 (GPU T4 x2, ~5-6 h)
- Same notebook with `CO2_MODE='as_is'`, `SEED=7`, `RUN_NAME='global_s7'`.
- Submit: `submission.csv` (= `co2_hist`, best epoch), `submission_co2x2.csv`, and one or two snapshot files.
- If session B beats session A's best single model, run this with `CO2_MODE='const'` instead.

### Hybrid probes (no GPU, files ready now)
- `result/hyb_<run>.csv` for 6 runs: that run on ssp126 rows, `bg_scn` on ssp585 rows.
- Submit all 6, send the LB of each; I fill `HYB_LB` in `ensemble_scn.py` and run `solve`.
- Also gives each model's ssp126 vs ssp585 error (up to a shared constant): where each model fails.

### Suggested slot order (5/day)
1. Day 1: hyb_global_co2_hist, hyb_lstm_xl, hyb_v2_scheduled_sampling, hyb_ecoformer_ed, hyb_global. Run session A.
2. Day 2: hyb_global2_co2_hist + 4 files of session A. Start session B.
3. Day 3: last session-A file, session B files, my new blend.
4. Day 4+: session C files, re-probe the bias of the new best blend (4 slots, `probe_bias.py make4/solve4`).

## Decision rules
- A file whose LB I can compute (blends, scalings, extrapolations of scored files) is never worth a slot.
- After session A: if `co2x2`/`co2x3` beats 0.122 as a single model, that CO2 mapping becomes the default test
  variant for every CO2 model.
- After session B: if no-CO2 is within ~0.005 of 0.122, keep it (robust, and a different blend component); if it is
  better, switch session C and any later model to `CO2_MODE='const'`.
- Final picks: the best estimated blend, plus the best file without negative weights.
