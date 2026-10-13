# Plan and runbook

## Now (2026-10-14, INSIGHTS 4.32)
1. Done: x2 = 0.063 (best), GBM y0inv2 = 0.097 (no blend gain).
2. Optional probe: `result/match_gbm_y0inv2.csv` (predicted 0.0955; add as `gbmm_y0inv2` in RUNS, re-solve x2's set).
3. Final picks: x2 (if it confirms) + x1 (0.066, scored) as the safer one; `submission_blend_x2h.csv` (est 0.0638,
   half the negative weights) is the in-between option.

## Earlier focus (2026-10-13): pairs + CO2-physical session F (INSIGHTS 4.28/4.29)

Best LB **0.066** (`result/submission_blend_x1.csv`, the extrapolation along the physical-LSTM pair; its
ssp585 agb level is confirmed by the cell probe, factor 1.005). Shared simplex blending is saturated.

1. GBM twin (local, done): submit `result/match_gbm_none.csv` (1 slot). Then add `gbmm_none` to ensemble.py RUNS and
   re-solve the x1 set + gbmm_none (free: lstm_xinv, and gbmm_none only if its LB is above ~0.16; INSIGHTS 4.30).
2. Session F done (`result/global-probe/`, co2inv 0.119). Next: submit `submission_global_y0inv_xinv_co2_hist.csv`
   (predicted 0.1215) and `submission_global_y0inv_xinv_co2inv0.5.csv` (predicted 0.1105), fill `g1_*` in RUNS,
   re-solve with the CO2 pair free (INSIGHTS 4.31). Old session F steps kept below for reference.
   **Session F (GPU T4 x1, ~1 h):** new Kaggle dataset `global-run1` from `kaggle_upload/global-run1/` (run-1 weights
   + reference preds). Notebook inputs: competition data, `zhihaow/carbonglobe`, `global-run1`, `noco2-ep4` (for
   test_maps.npz; it also holds snap_ep4.npz, which is ignored in this profile). `global_probe.py` as committed
   (PROFILE='co2phys'). Check the log line `reproduction ...: msd(co2_hist, original run)` is ~1e-5 (0.07 = wrong
   weights: stop). Submit `submission_global_y0inv_xinv_co2inv.csv`, `submission_global_xinv_co2inv.csv`,
   `submission_global_y0inv_xinv_co2_hist.csv`, then `submission_global_y0inv_xinv_co2inv0.5.csv`;
   send `global_probe_preds.npz`. Then `ensemble_scn.py ingest <run> <npz> <key> map` and re-solve with pairs free.
3. Final picks: best estimated blend (negative weights allowed, with LB confirmation) + `x1h` or `e4` (safer).
4. Do not submit `result/probe_co2_adj.csv` (eda5.py: wrong CO2 premise, wrong direction).

## Done (2026-10-12): own-site analogs (INSIGHTS 4.27/4.28)
- `analog.py` add 0.173, hist 0.209: weight 0. Session E LSTM y0inv 0.111, y0inv_xinv 0.109, xinv 0.196; e4 0.068.

## Earlier focus (2026-10-07): generalization

1. **Overfit check on the public LB** (`cv_parity.py`, site parity). Submit `result/par_<run>.csv` for noco2_ep4,
   noco2_ep3, noco2_ep6, s7_ep3, v2_scheduled_sampling, lstm_xl (6 slots); fill `PAR_LB`; `python cv_parity.py solve`.
   Answers: is noco2_ep4's lead real in both halves, how much does picking the best snapshot cost (winner's curse),
   how much do fitted blend weights overfit.
2. **Snapshot + seed averaging** (`carbonglobe-global.ipynb`, defaults: no CO2, SEED 11, `global_noco2_s11`,
   snapshots 2–6). Submit `submission_snapmean_none.csv` first (no selection), then single snapshots only if slots
   are free. Cross-seed averages: `python cv_parity.py average <runs> <tag>`.
3. **Final bias correction with CV**: on the final blend, `make_bias` (8 probes: column × scenario × half), then
   `solve_bias` gives the pooled per-scenario factors plus an in-sample vs cross-fitted gain.
4. **Final picks**: best fitted blend + bias (no negative weights), and the zero-parameter average + bias.

### Update (INSIGHTS 4.22/4.23): the test is the grid behind affine maps; the truth is in the mapped space (T2)
- probe_dead_T2.csv = **0.072** (best). Every model now: map test inputs back to physical, run, map outputs forward.
- **Kaggle session E** (`global_probe.py` as committed, GPU T4 x1, no repo clone): add `test_maps.npz`
  (kaggle_upload/noco2-ep4/, written by test_transform.py) to the `noco2-ep4` dataset as a new version. Inputs:
  competition data, `zhihaow/carbonglobe`, `noco2-ep4` (snap_ep4.npz + test_maps.npz). Variants none, y0inv,
  y0inv+xinv, xinv. Submit `submission_global_y0inv.csv`, then `submission_global_y0inv_xinv.csv`; send
  `global_probe_preds.npz` (physical predictions; locally `ensemble_scn.py ingest <run> <npz> <key> map`).
- Local: `carbonglobe_gbm.py` (LightGBM 4.6.0, data/ folder) -> result/gbm/.
- Candidate now: `result/submission_blend_scn2d_x.csv` (est 0.0733 -> ~0.071).

### Update (INSIGHTS 4.21): tree models as diverse blend members
- Averaging more grid LSTMs (subsets of rows/features/seeds) converges to ~0.097: no gain. A new member needs
  msd(member, 0.074 blend) > LB(member) − 0.074; target ~0.11 LB with error correlation ≤ ~0.65.
- Run `carbonglobe_gbm.py` on Kaggle: **CPU notebook** (no GPU quota), inputs = competition data +
  `zhihaow/carbonglobe`, no repo clone, no Internet needed. Paste into one cell, Save & Run All (~2–4 h).
- Submit `submission_gbm_direct.csv` and `submission_gbm_rollout.csv`; send both LBs + `gbm_preds.npz` +
  `gbm_summary.json`. Then: `python ensemble_scn.py ingest gbm_direct gbm_preds.npz direct` (same for `rollout`),
  2-way estimate vs the 0.074 blend, hybrids only for members that pass, `blend_scn4`.
- `_dsh_v2` files only if the plain ones get weight.

### Update (INSIGHTS 4.20): y0 done, plan closed
- `submission_y0.csv` 0.285, `hyb_y0_persist.csv` 0.174 → y0 gets weight 0.03–0.08; per-scenario blend estimate
  0.0759 (scn2: 0.0764) and 0.0752 with negative weights (scn2_x: 0.0753). Gain ≤ 0.0005 = LB rounding: not submitted.
- **Final picks: `result/submission_blend_scn2.csv` + `result/submission_blend_scn2_x.csv` (both 0.074).**

### Update 2026-10-11 (INSIGHTS 4.19): stop training; y0 as a blend component
- swa_ep4 scored 0.136; historical holdout is anti-correlated with the LB (−0.85). No more training runs.
- Session D done: no single feature beats `dsh_v2`; v3, f7, static are not used by the model.
- Next (CPU, 2 slots): `y0_persist.py` in a Kaggle cell (competition data only) → submit `submission_y0.csv`, send
  `y0_preds.npz` → I build `result/hyb_y0_persist.csv` (1 slot) → per-scenario blend with y0.
- Final picks unless that wins: `submission_blend_scn2.csv` + `submission_blend_scn2_x.csv` (both 0.074).

### Session D: feature selection without retraining (INSIGHTS 4.18, done)
- Add `global_noco2_holdout.npz` to the `noco2-ep4` dataset (new version; both files are in `kaggle_upload/noco2-ep4/`).
- Notebook: one cell with `global_probe.py` as committed (parts A/B/C off, D1 + D2 on). Inputs: competition data,
  `zhihaow/carbonglobe`, `noco2-ep4`.
- Send back the `== D1` table, the `== D2` table and the per-group sum lines (or `global_probe_importance.csv` and
  `global_probe_feature_screen.csv`). I then pick 1–3 `dshf_...` variants (1 slot each) for part C.
- Can run independently of the `swa_ep4` training session.

### Update 2026-10-10 (INSIGHTS 4.17): fine-tune around the 0.091 model
- global_nov2 (v2 dropped in training) scored 0.135: superseded. From-scratch no-CO2 runs typically score 0.11–0.13.
- Next GPU run = `carbonglobe-global.ipynb` as committed, profile `swa_ep4`: initialised from `snap_ep4.npz` (attach the
  `noco2-ep4` dataset, the same one as for `global_probe.py`), constant LR 5e-5, 8 short epochs of 0.5M samples,
  snapshots 2/4/6/8. GPU T4 x2, Internet on, competition data + `zhihaow/carbonglobe` + `noco2-ep4`, no checkpoint.
- Check in the log: `initialised from .../snap_ep4.npz | lr 5.0e-05 (constant ...)`; epoch-1 holdout shift should be
  close to run 3's epoch-4 value (0.038). If it is far worse, stop the session.
- Submit `submission_ep2_none.csv` first (closest to the 0.091 weights: tells whether fine-tuning keeps the quality),
  then `submission_snapmean_none.csv`; send `swa_ep4_preds.npz`.

### Update 2026-10-09 (INSIGHTS 4.14)
- Spatial bagging dropped: bag_m1 (70% of blocks) scored 0.128 and gets weight 0 in every blend.
- Next GPU run = `carbonglobe-global.ipynb` as committed: `global_nov2` = no CO2 + group v2 held at its mean, all
  sites, SEED 42 (same seed as the 0.091 run, so the only difference is v2). Submit `submission_snapmean_none.csv`
  first, then `submission_ep4_none.csv`; send `global_nov2_preds.npz`.
- After that, more diversity = more seeds of the better of {no CO2, no CO2 + no v2}, full data, snapmean only.

### Ensemble of bagged members (2026-10-08, superseded: bagging hurt)

One Kaggle session = one member: `carbonglobe-global.ipynb`, change only the 4 lines below, no checkpoint attached.
Submit only `submission_snapmean_none.csv` per member (1 slot); send `<RUN_NAME>_preds.npz`.
Final = equal-weight mean of all members' snapmean (+ earlier no-CO2 runs), then per-scenario bias via `cv_parity.py`.

| Member | RUN_NAME | SEED | BAG_FRAC | DROP_GROUPS | Purpose |
|---|---|---|---|---|---|
| m1 | bag_m1 | 21 | 0.7 | () | default as committed |
| m2 | bag_m2 | 22 | 0.7 | () | |
| m3 | bag_m3 | 23 | 0.7 | from screening of no-CO2 ep4 (e.g. ('v2',)) | invariance to a shifted group |
| m4 | bag_m4 | 24 | 0.7 | () | |

Spatial blocks are 10°×10° (`BAG_BLOCK_DEG`); the holdout is identical for every member (`HOLDOUT_SEED`).

### Data probe of the best model
- Session A (parts A + B): **done** (INSIGHTS 4.13). Only feature group v2 moves the no-CO2 model; the year-40
  prediction keeps 55–62% of a shift in y0.
- **Session A2** (part C, ~1 h GPU T4 x1): `global_probe.py` as it is now (parts A/B off, `FULL_VARIANTS = none,
  dsh_v2, y0fx1, y0bx1`). Inputs: competition data, `zhihaow/carbonglobe`, the `noco2-ep4` dataset
  (`kaggle_upload/noco2-ep4/snap_ep4.npz`); optional `result/globel-train/third-run/global_noco2_preds.npz` for the
  reproduction line. `submission_global_none.csv` = the scored 0.091 file: do not submit it.
  Submit `submission_global_dsh_v2.csv`, `submission_global_y0fx1.csv`, then `submission_global_y0bx1.csv`.

| Result (vs 0.091) | Meaning | Next |
|---|---|---|
| `dsh_v2` better | the model's response to the v2 shift is spurious, like CO2 was | member m3 with `DROP_GROUPS=('v2',)` |
| `dsh_v2` worse | the v2 shift is real signal | keep v2 in every member |
| `y0fx1` better | initial fluxes mislead the model | a training option that hides lai/gpp/npp/rh of the *initial* state |
| `y0fx1` worse | initial fluxes carry useful information (likely the present-day CO2 effect) | keep |
| `y0bx1` much worse | stock memory is real physics (expected) | nothing; it is a control and a blend component |

Every scored variant is also a blend component (same model, different treatment of one shifted input), like the
`global` / `global_co2_hist` pair that gave the negative-weight gain.

Older runbook below.

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
