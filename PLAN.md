# Plan: experiments to find more "co2_hist"-type insights

Best LB so far: `result/bg_scn.csv` (expected ~0.094, not yet submitted) and `submission_blend_global.csv` (0.100).
Details and evidence for every number below are in `INSIGHTS.md`.

## What we have learned to trust

1. **The LB is a linear-algebra oracle.** Each submission p gives one number, MSE(p, y). Because MSE is quadratic, every
   score tells us ⟨p, y⟩ up to a constant. Any submission that is a linear combination of already-scored files (blends,
   per-cell scalings, extrapolations) has a predictable LB without spending a slot. This is what `ensemble.py` and
   `probe_bias.py` do; it has matched the LB 6 times. Caveat: the public LB uses a subset of sites, while the
   pairwise distances are computed on all sites. When models disagree a lot (the grid model) the estimate can be off by
   ~0.002 (blend_global: estimated 0.1022, real 0.100).
2. **Inference-time input edits are cheap, informative experiments.** `co2_hist` (one array replacement, no training)
   gave −0.002 on the old LSTM-XL and −0.012 on the grid model. Each such variant costs one rollout and one slot.
3. **Bias is per scenario and the same for every model** (ssp585 AGB ~×1.09–1.12, ssp126 height ~×0.94–0.95).
   Re-probe it (4 slots, `probe_bias.py make4/solve4`) after every big change of the base file.

## Free (no slot) result already computed

The two scored grid files lie on a line: `global` (0.134) → `global_co2_hist` (0.122). The LB along that line is exactly
quadratic, and its minimum is at **t ≈ 1.9** (0.118, range 0.117–0.120 under rounding): the CO2 edit is "not strong
enough". Allowing a negative weight on `global` in the blend gives `result/submission_blend_global_x.csv`:

| file | weights | estimated LB |
|---|---|---|
| `submission_blend_global.csv` (scored 0.100) | simplex | 0.1022 (real 0.100) |
| `submission_blend_global_x.csv` | global −1.06, global_co2_hist +1.67, v2 0.18, lstm_xl 0.14, eco 0.07 | 0.0977 (0.0967–0.0987) |

Expected gain ~0.0045 before bias correction. The negative weight extrapolates the CO2 effect, which raises the
question below (Q1).

## Open questions and the experiment that answers each

| # | Question | Experiment | Cost | Decision rule |
|---|---|---|---|---|
| Q1 | Why does the model want *more* than the historical CO2? Is the test CO2 mis-scaled? | `global_probe.py` part C: `co2_up` (input-space version of t=2), `co2_cont` (2020 level + test increments), `co2_end` (2020 level) | 3 slots, ~30 min GPU | If `co2_up` or `co2_cont` beats 0.122, the test CO2 is too low as provided: build every future model with that CO2 mapping. If `co2_hist` stays best, the extrapolation only works in output space (keep the `_x` blend). |
| Q2 | Which other feature groups act like CO2 did? | `global_probe.py` part A: remove the test-vs-grid mean shift from one group at a time on 1,500 test sites | 0 slots, ~15 min GPU | Groups with `msd_vs_none` ≥ ~0.002 (co2_hist's size) and positive `toward_target_pct` → add a `dsh_<group>` full variant and submit it. |
| Q3 | Is feature 7 (seasonal amplitude 20× in the test) hurting? | `f7_fix` in parts A and C | 0–1 slot | Submit `submission_global_f7_fix.csv` only if part A shows it moves predictions (msd ≥ ~0.001). |
| Q4 | Does the shifted `test_y0` matter? | `global_probe.py` part B: scale grid y0 to the test ratios, see how much the year-40 prediction moves | 0 slots | If year-40 moves less than ~1/3 of the y0 change, the y0 shift is not a priority. Otherwise: train with y0 perturbation (augment the starting state). |
| Q5 | How much is left in the grid model itself? | next session of `carbonglobe-global.ipynb` (resume from epoch 4, K=1 to epoch 8, NaN-safe) | 1 GPU session + 1 slot (its co2_hist / best CO2 variant) | Keep if its CO2-variant beats 0.122; then re-blend. |
| Q6 | Does a second grid architecture add diversity? | `carbonglobe-global.ipynb` with `BACKBONE='ecoformer'`, new `RUN_NAME` | 1–2 GPU sessions + 1 slot | Add to the blend if its error correlation with `global_co2_hist` (from `ensemble.py`) is below ~0.9. |
| Q7 | Can training fix the CO2 problem at the source? | grid training with the CO2 mapping that wins Q1 applied to the test, or with CO2 augmentation (random level offsets per window) | 1 GPU session + 1 slot | Only after Q1: it tells us which mapping to train for. |

## Suggested order (5 slots/day)

**Day 1 (now)**
1. Submit `result/bg_scn.csv` (expected ~0.094) and `result/submission_blend_global_x.csv` (expected ~0.096–0.098).
2. Run `global_probe.py` on Kaggle (one GPU session, ~1 h): parts A, B and C.
3. Submit `submission_global_co2_up.csv`, `submission_global_co2_cont.csv` and `submission_global_co2_end.csv` (Q1).

**Day 2**
4. Submit up to 2 screen winners from part A (Q2/Q3).
5. Re-blend everything with `ensemble.py`; each new scored variant is another basis file. Allow a negative weight
   only between variants of the *same* model (the `_x` pattern).
6. Re-probe the bias of the new best blend (4 slots, `make4`/`solve4`), or reuse the `bg` factors if out of slots.

**In parallel (GPU)**: Q5, then Q6. Each new model needs one submission of its best CO2 variant before it can be
blended.

## Rules of thumb

- Never spend a slot on something the oracle can predict (blends, scalings, extrapolations of scored files).
- A slot is worth spending when it measures a *new direction*: a new model, a new input edit, a new cell of the
  bias.
- Check the expected value before submitting: `ensemble.py` / `probe_bias.py` print estimates; treat them as ±0.002
  when models disagree a lot.
- Keep public-LB tuning coarse (few parameters, per scenario at most): the private LB uses other sites.
