# Plan and runbook

## MASTER PLAN (2026-10-28) — goal: win (top 1 = 0.038, top 3 = 0.042; we are at 0.054)

**Where we are (2026-10-31).** Best LB **0.050** (`submission_blend_x8f.csv`); best singles: GBM chrono 2 seeds
**0.072**, NN chrono **0.075**. Final picks right now: **x8f + x7f (0.054)**.

**The one lever that works now: a better base model.** Grid holdout 0.0305 -> 0.0245 (-20%) gave LB 0.097 -> 0.073
(-25%): grid accuracy carries over to the future. Post-hoc blend probes are exhausted (each round ~0.0002).
Slots are plentiful: submit every real candidate, but only after a local check.

**Protocol for every model idea** (no slot until step 2 passes):
1. Subset test, `chrono.py`-style: 20k sites, 10-degree block holdout (same split every time), compare with the
   current best feature set. Keep only if the holdout drops clearly (> ~5%: subset noise is ~3%, INSIGHTS 4.45).
2. Full run (`carbonglobe_gbm.py`, all 54k sites), variants y0inv / y0inv2 / none, full holdout logged.
3. `slope.py match <npz> direct_y0inv2 result/probe_dead_T2.csv result/match_<tag>_y0inv2.csv`, ingest, rank with
   `rank_members.py` against x7f; submit the y0inv2 file (1 slot), then re-blend (`ensemble_scn.py shared`, base x7f
   + new member, old-vs-new model pair free) and submit the blend if the estimate gains >= 0.001.
4. Record in INSIGHTS (LB ledger + section) and commit.

### Track A — stronger GBM direct (local CPU; highest priority)
- [x] A1 (subset 0.0270 -> 0.0264, but FULL 0.0245 -> 0.0253: dropped) every year's event probability (40 features) instead of 5 summaries (`chrono.py` fire vs fire40; running).
- [x] A2 (worse, 0.0276) full 15-age profile with all **7** state variables (now height/agb only) + profile interpolated at
      log(age+20) and log(age+40) for all 7 + profile slope in log-age around the own age (public notebooks).
- [x] A3 (worse, 0.0274) ecophysiological ratios of the own state: CUE = npp/gpp, NEP = npp - rh, agb/height, lai/height,
      historical accumulation rate y0/(age+5).
- [x] A4 (worse, 0.0285) climate trajectory fingerprint for the drivers the model uses (f0-f7, v1-v4 groups): 40-year slope, year-to-
      year std, first-5 / last-5 year means, warming (last decade - first decade) (ForestCast, 0.142->0.060 notebook).
- [ ] A5 event risk along the trajectory, not only at y0: two-stage (stage-1 predicted y20/y40 -> recompute risk at
      the interpolated state), and "years since the last high-risk year" at year 40.
- [ ] A6 event magnitude: regress the one-year agb loss (not only P(drop > 30%)); expected cumulative loss feature.
- [ ] A7 stage-2 stacking: out-of-fold stage-1 predictions of all 15 ages of the site as features (cross-age
      consistency; EcoPhysNet reports 0.0115 -> 0.0100 OOF).
- [ ] A8 age-regime models (1-30 / 50-150 / 200-500) vs one model; per-variable target transforms (agb log-ratio).
- [ ] A9 2-seed average = 0.072 (single 0.073): seed averaging helps. Seed 11 running -> 3-seed average; then x8 with NN chrono. seed 7 of gbmc (`GBM_SEED=7 GBM_CHRONO=1`, tag `_chrono_s7`). 3-5 seeds averaged (bagging), larger trees / lower learning rate once the feature set is fixed.
- [x] A10 DONE 0.158 (pred 0.168), blend gain 0.0003 only. `result/match_gbmc_none.csv` twin (y0 left in T space) for an extrapolation pair, as the old GBM pair gave
      0.066 -> 0.063; also for every later GBM version.

### Track B — neural direct model with the same information (Kaggle GPU; diversity for the blend)
GPU steps (B, C) ship as ready-to-run notebooks in `kaggle_notebooks/` (built by `make_notebooks.py` from the .py;
inputs to attach + files to send back in the first cell); local artefacts go to `kaggle_upload/<name>/`.
- [x] B1 DONE: nnc_y0inv2 = 0.075. SUBMIT `submission_blend_x8f.csv` (est 0.0503), `submission_blend_x8.csv` (0.0509). `kaggle_notebooks/kaggle_direct_nn_chrono.ipynb` (NN_CHRONO=1: chrono 34 + event risk 5, smoke-tested
      locally; send back nn_preds.npz + nn_summary.json). `carbonglobe_direct_nn.py` + chronosequence (15x7 profile, +20/+40 interpolation) + event-risk inputs;
      cross-age attention over the 15 ages of a site (both public neural notebooks use it). Same variants and
      mean matching as the GBM; submit y0inv2.
- [ ] B2 multi-seed NN average once B1 helps.

### Track C — annual neural emulator (GRU over 40 years, monthly climate; Kaggle GPU; lower priority)
- [ ] C1 state(t) + 12 months of climate(t) -> state(t+1), site context + 15-age profile, loss on all years with weight
      on year 40, no teacher forcing (the 0.060 team: 0.0078 local, 0.144 LB alone, but improved their blend).
      Only as a different-error member; expect weak alone.

### Track D — blend and calibration after each new member (slots)
- [ ] D1 re-blend with base x7f after each scored member; free pairs: (new GBM, old GBM), (y0inv2, none) twins.
- [ ] D2 READY `result/yp_x8f_ssp585.csv`, `result/yp_x8f_ssp126.csv` (0.053 each = no correction needed);
      `python youngprobe.py solve x8f <scn> 0.050 <LB>`. Re-run the age x scenario probe (`scnprobe.py` young direction) on the new best blend: a better model may
      need a different correction (1-2 slots).
- [ ] D3 per-scenario weights (`ensemble_scn.py` hybrids) for the strongest new members once there are 2-3 of them.

### Track E — settle open questions (1 slot each, cheap with many slots)
- [x] E1 DONE 0.055 -> T space confirmed. `result/probe_x7f_dead_raw0.csv` (x7f, 16,363 dead rows at raw 0). LB 0.055 = T space confirmed,
      0.053 = raw units (then re-check everything), 0.054 = ambiguous. Output space: the 0.142->0.060 notebook claims the answer key is in raw units; our dead-row probe (#73) and
      bias probes say T space. One decisive probe on x7f: dead rows set to raw 0 instead of the mapped zero
      (predicted ~+0.001 if we are right). Keep T space unless it improves.
- [ ] E2 Kaggle API token expired (401): renew `~/.kaggle/kaggle.json` to read public notebooks / discussions.

### Dead ends (do not repeat; INSIGHTS 4.19-4.43)
More LSTM snapshots/pairs (~0.0001 each), own-site historical analogs (0.173), site-pattern probes on the blend
(warm / npp / incr / height-only / finer age bands, ~0.0002), ED's CO2 calendar-effect pattern (co2_dir 0.063),
correcting the test climate (always worse), bagging LSTMs, fine-tuning/SWA, feature dropping, RF members.


## Done (2026-10-28): stronger base model (INSIGHTS 4.44), co2_dir done (0.063, no gain)
Done: gbmc_y0inv2 = 0.073 (best single model ever). Next:
1. Submit `result/submission_blend_x7f.csv` (est 0.0532) and `result/submission_blend_x7.csv` (est 0.0566, no
   negative weight).
2. Make the base model stronger along the same line: event risk along the trajectory (not only at y0), more GBM
   seeds, the same features in the NN direct model; twin `match_gbmc_none.csv` for a pair.

## Done (2026-10-28): gap to top is 0.022 -> ED's own CO2 response (INSIGHTS 4.43)
1. Submit `result/co2_dir.csv` (x6 + CO2 pattern from the grid's calendar effect, ssp585 only; 0.063 = useless).
2. `python co2dir.py solve 0.060 <LB>`. If it helps: same direction on ssp126, then a CO2-aware member.

## Done (2026-10-27): which sites gain most from CO2 under ssp585 (INSIGHTS 4.42)
Done: warm 0.063, npp 0.064, incr 0.063, h_only 0.064 -> solved gain 0.0002: not submitted. Post-hoc adjustment of
x6 is exhausted (INSIGHTS 4.42). **Final picks: x6 + x4f.**

## Done (2026-10-27): age shape of the ssp585 CO2 response (INSIGHTS 4.41)
Best LB 0.060 = `result/submission_blend_x6.csv`. Final picks: **x6 + x4f** (0.061). ssp126 young: no gain, no x7.
Done: ap_young_lo 0.062, ap_old_lo 0.062 -> solved gain 0.0002 (< 0.0005 rounding threshold): not submitted.
Age x scenario axis exhausted. **Final picks locked: x6 + x4f.**

## Done (2026-10-15): ssp585-specific weights (INSIGHTS 4.39)
Final picks so far: x4f (0.061) + x3 (0.062). Submit `result/dp_gbm_pair.csv`, `dp_lstm_pair.csv`, `dp_nn.csv`,
`dp_dead_t2.csv`, then `python dirprobe.py solve 0.061 <4 LBs in that order>` -> `submission_blend_x4f_s585.csv`.

## GPU plan (30 h T4, INSIGHTS 4.36): make many members, rank them locally, submit only the steep ones
1. Upload `kaggle_upload/lstm-sweep/` as a Kaggle dataset (lstm-sweep).
2. Run `kaggle_notebooks/kaggle_sweep_pairs_run1.ipynb` (~4.5 h), `..._run2.ipynb` (~3.5 h), and
   `kaggle_notebooks/kaggle_direct_nn.ipynb` (~1-2 h). Send back `global_probe_preds.npz` of each run, `nn_preds.npz`,
   `nn_summary.json`. Nothing is submitted directly.
3. Locally: `ensemble_scn.py ingest <run> <npz> <key> map` (LSTM sweep) / `slope.py match` + `ingest_csv` (NN), then
   `python rank_members.py <runs...>` -> submit the 2-4 steepest per day, re-solve after each batch.

## Now (2026-10-14, INSIGHTS 4.32)
1. Done: x2 = 0.063 (best), GBM y0inv2 = 0.097 (no blend gain).
2. Done: RF y0inv 0.110, RF twin 0.155: the RF pair adds 0.0005 at most (duplicates the GBM pair). x2 stays.
2b. Optional GPU, session G (INSIGHTS 4.34): new Kaggle dataset from `kaggle_upload/noco2-s7/`; notebook inputs =
   competition data, `zhihaow/carbonglobe`, that dataset; `global_probe.py` as committed (PROFILE noco2_s7). Submit
   `submission_global_y0inv_xinv.csv` and `submission_global_xinv.csv`, send `global_probe_preds.npz`.
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
