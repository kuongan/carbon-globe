"""Build the Kaggle notebooks in kaggle_notebooks/ from the committed scripts (the .py files stay the source of truth).

usage: python make_notebooks.py
"""
import json, os

OUT = 'kaggle_notebooks'

SWEEP_DOC = """# LSTM pair sweep, run {run} (inference only, INSIGHTS 4.36)

Rolls out existing no-CO2 LSTM-XL snapshots on the test set with y0 + forcing mapped to physical units
(`y0inv+xinv`) and with forcing only (`xinv`): one "pair" per snapshot, like the pair behind x1 / x2.

- **Accelerator:** GPU T4 (one GPU is used). **Internet:** off is fine. Expected time: {hours}.
- **Inputs:** competition data, `zhihaow/carbonglobe`, and the dataset made from `kaggle_upload/lstm-sweep/`
  (`sweep_<tag>.npz` weights, `ref_<tag>.npz` reproduction references, `test_maps.npz`).
- This run: `SWEEP_ONLY = {only}`.
- **Check in the log:** one `reproduction <tag>: msd(none, scored file) ... = ~1e-6` line per snapshot; the run stops
  with an assertion if a snapshot does not reproduce its scored predictions.
- **Send back:** `global_probe_preds.npz` (saved after every snapshot, so a timeout keeps the finished ones).
  Do not submit anything yet: the CSVs are ranked locally first.
"""

NN_DOC = """# Neural direct model (training, INSIGHTS 4.36)

A new blend-member family: y0 + 40 years of climate -> year-40 state in one step (like the LightGBM direct
model, but a neural net). 4 seeds, averaged; 5 test variants (y0inv, none, y0inv2, inv, xinv).

- **Accelerator:** GPU T4 (one GPU is used). **Internet:** off is fine. Expected time: ~1-2 h.
- **Inputs:** competition data, `zhihaow/carbonglobe`, and a dataset holding `test_maps.npz`
  (`noco2-ep4` or the lstm-sweep dataset).
- **Check in the log:** `pre-flight ok` within ~15 min (data loading ~10 min, then 20 training steps and every
  test variant on 13 sites); then one `holdout year-40 metric` line per epoch (persistence is ~0.37).
- **Send back:** `nn_preds.npz` and `nn_summary.json`. Do not submit anything yet: mean-matched files are made
  locally.
"""


NN_CHRONO_DOC = """# Neural direct model + chronosequence + collapse-event risk (training, INSIGHTS 4.44, PLAN track B1)

Same neural direct model as `kaggle_direct_nn`, plus the two inputs that took the LightGBM direct model from LB 0.097
to 0.073: the site's own chronosequence (y0 of all 15 seed ages + y0 at age a+40) and the risk of site-year biomass
collapses (a LightGBM event classifier trained inside this notebook). 4 seeds, averaged; variants y0inv2, none, y0inv.

- **Accelerator:** GPU T4 (one GPU is used). **Internet:** off is fine. Expected time: ~2.5-3.5 h
  (data ~10 min, event classifier ~10 min, risk features for the grid and 6 test variants ~1 h on CPU, 4 seeds x 20
  epochs ~1-1.5 h).
- **Inputs:** competition data, `zhihaow/carbonglobe`, and a dataset holding `test_maps.npz` (`noco2-ep4` or the
  lstm-sweep dataset), exactly as for `kaggle_direct_nn`.
- **Check in the log:** `event classifier: ... event rate ~0.013`; `chrono + risk features (grid) .../...` lines;
  `pre-flight ok`; then one `holdout year-40 metric` line per epoch. The old NN reached ~0.029; the LightGBM with
  these features 0.0245: anything clearly below 0.029 is a success.
- **Send back:** `nn_preds.npz` and `nn_summary.json` (saved after every seed: a timeout keeps the finished seeds).
  Do not submit anything: mean-matched files are made locally.
"""


NN_XAGE_DOC = """# Neural direct model, cross-age version (training, PLAN track B2)

The chrono NN (`kaggle_direct_nn_chrono`, LB 0.075, the blend x8f = 0.050) with one change: the 15 seed ages of a
site go through the net together, and a 2-layer transformer lets every age see the others (canopy closure and
competition along the site's chronosequence). New seeds 4-7, so it also adds seed diversity. Variants y0inv2, none,
y0inv.

- **Accelerator:** GPU T4 (one GPU is used). **Internet:** off is fine. Expected time: ~2.5-3.5 h (same data and
  risk-feature steps as the chrono notebook, then 4 seeds x 20 epochs).
- **Inputs:** exactly as for `kaggle_direct_nn_chrono`: competition data, `zhihaow/carbonglobe`, and the dataset
  holding `test_maps.npz`.
- **Check in the log:** `event classifier: ... event rate ~0.013`, `pre-flight ok`, then one `holdout year-40 metric`
  line per epoch. The per-row chrono NN reached 0.0243-0.0263 per seed: lower is better, similar is still useful.
- **Send back:** `nn_preds.npz` and `nn_summary.json` (zip them as before). Do not submit anything.
"""


NN_XAGE2_DOC = """# Neural direct model, cross-age version 2 (training, PLAN track B3)

The cross-age NN (`kaggle_direct_nn_xage`) scored **0.061** alone, the best single model. Same model, bigger and
trained longer: token width 256 (was 192), 3 transformer layers (was 2), 30 epochs (was 20), new seeds 8-11 (a
second set of seeds also averages out seed noise, as with the GBM: 0.073 -> 0.072).

- **Accelerator:** GPU T4 (one GPU is used). **Internet:** off is fine. Expected time: ~4-5.5 h.
- **Inputs:** exactly as for `kaggle_direct_nn_xage`.
- **Check in the log:** `pre-flight ok`, then `holdout year-40 metric` per epoch. Version 1 reached 0.0239-0.0256
  per seed.
- **Send back:** `nn_preds.npz` and `nn_summary.json`, zipped as `kaggle_direct_nn_xage2.zip`. Do not submit anything.
"""


NN_MONTHLY_DOC = """# Neural direct model, cross-age + monthly climate (training, PLAN track B4)

Cross-age NN v2 (d_tok 256, 3 layers, 30 epochs; v1 alone LB 0.061) plus one new input: a small conv over the 12
months of f0-f6 of every year (mean / max / min over the months), appended to the 40-year sequence. The collapse
events depend on monthly extremes; until now the net only saw 5 risk summaries of them. New seeds 12-15.

- **Accelerator:** GPU T4 (one GPU is used). **Internet:** off is fine. Expected time: ~4.5-6 h.
- **Inputs:** exactly as for `kaggle_direct_nn_xage` / `xage2`.
- **Check in the log:** `pre-flight ok`, then `holdout year-40 metric` per epoch. v2 reached 0.0229-0.0242 per seed:
  clearly lower = the monthly input helps.
- **Send back:** `nn_preds.npz` and `nn_summary.json`, zipped as `kaggle_direct_nn_monthly.zip`. Do not submit anything.
"""


def notebook(markdown, code):
    cell = lambda kind, src, **kw: dict(cell_type=kind, metadata={}, source=src.splitlines(keepends=True), **kw)
    return dict(nbformat=4, nbformat_minor=4,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python')),
                cells=[cell('markdown', markdown), cell('code', code, execution_count=None, outputs=[])])


def write(name, nb):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('wrote', path)


probe = open('global_probe.py', encoding='utf-8').read()
assert "PROFILE = 'sweep'" in probe, 'global_probe.py must be committed with PROFILE = sweep'
DEFAULT_ONLY = "SWEEP_ONLY = ('s7ep3', 's42ep6', 'bagep6', 's42ep3')"
assert DEFAULT_ONLY in probe
for run, only, hours in ((1, ('s7ep3', 's42ep6', 'bagep6', 's42ep3'), '~4.5 h (4 snapshots x ~65 min)'),
                         (2, ('s7ep5', 'swaep4', 's42ep5'), '~3.5 h (3 snapshots x ~65 min)')):
    code = probe.replace(DEFAULT_ONLY, f'SWEEP_ONLY = {only!r}')
    write(f'kaggle_sweep_pairs_run{run}.ipynb', notebook(SWEEP_DOC.format(run=run, only=only, hours=hours), code))

nn_code = open('carbonglobe_direct_nn.py', encoding='utf-8').read()
write('kaggle_direct_nn.ipynb', notebook(NN_DOC, nn_code))
_chr = "CHRONO = os.environ.get('NN_CHRONO', '0') == '1'"
assert _chr in nn_code
chrono_code = nn_code.replace(_chr, "CHRONO = os.environ.get('NN_CHRONO', '1') == '1'")
write('kaggle_direct_nn_chrono.ipynb', notebook(NN_CHRONO_DOC, chrono_code))
_xa = "XAGE = os.environ.get('NN_XAGE', '0') == '1'"
assert _xa in chrono_code
xage_code = chrono_code.replace(_xa, "XAGE = os.environ.get('NN_XAGE', '1') == '1'")
write('kaggle_direct_nn_xage.ipynb', notebook(NN_XAGE_DOC, xage_code))
x2 = xage_code
for old, new in (("os.environ.get('NN_DTOK', 192)", "os.environ.get('NN_DTOK', 256)"),
                 ("os.environ.get('NN_XLAYERS', 2)", "os.environ.get('NN_XLAYERS', 3)"),
                 ("os.environ.get('NN_EPOCHS', 20)", "os.environ.get('NN_EPOCHS', 30)"),
                 ("'4,5,6,7' if XAGE", "'8,9,10,11' if XAGE")):
    assert old in x2, old
    x2 = x2.replace(old, new)
write('kaggle_direct_nn_xage2.ipynb', notebook(NN_XAGE2_DOC, x2))
xm = x2
for old, new in (("os.environ.get('NN_MONTHLY', '0')", "os.environ.get('NN_MONTHLY', '1')"),
                 ("'8,9,10,11' if XAGE", "'12,13,14,15' if XAGE")):
    assert old in xm, old
    xm = xm.replace(old, new)
write('kaggle_direct_nn_monthly.ipynb', notebook(NN_MONTHLY_DOC, xm))


GBM_ANCHOR_DOC = """# LightGBM direct, chronosequence-anchored target (CPU, final push for 0.044)

The chrono GBM (`GBM_CHRONO=1`, LB 0.073 alone) predicts `y40 - own y0`. Here the target is
`y40 - (the site's own y0 at age a+40)`: on the grid that anchor alone is already close for young stands (age 1:
0.037 vs persistence 0.93), and the test provides the same 15-age profile per site. The trees then learn only the
correction, so less of the level depends on extrapolating the historical dynamics. Same features, seed 42 (same
holdout as the chrono GBM, 0.0245), variants y0inv / y0inv2 / none.

- **Accelerator:** none (CPU). **Internet:** off is fine. Expected time: ~45-75 min on Kaggle CPU.
- **Inputs:** competition data, `zhihaow/carbonglobe`, and a dataset holding `test_maps.npz` (e.g. `noco2-ep4`).
- **Check in the log:** `event classifier ...`, then `direct holdout metric X` (chrono GBM without the anchor:
  0.0245 on this same holdout: lower is good).
- **Send back:** `gbm_preds_anchor.npz`, `gbm_summary_anchor.json` (zip as `kaggle_gbm_anchor.zip`). Do not submit.
"""
gbm_code = open('carbonglobe_gbm.py', encoding='utf-8').read()
for old, new in (("os.environ.get('GBM_MODELS', 'direct,rollout')", "os.environ.get('GBM_MODELS', 'direct')"),
                 ("os.environ.get('GBM_TAG', '')", "os.environ.get('GBM_TAG', '_anchor')"),
                 ("os.environ.get('GBM_VARIANTS', 'inv,y0inv')", "os.environ.get('GBM_VARIANTS', 'y0inv,y0inv2,none')"),
                 ("os.environ.get('GBM_CHRONO', '0')", "os.environ.get('GBM_CHRONO', '1')"),
                 ("os.environ.get('GBM_ANCHOR', '0')", "os.environ.get('GBM_ANCHOR', '1')")):
    assert old in gbm_code, old
    gbm_code = gbm_code.replace(old, new)
write('kaggle_gbm_anchor.ipynb', notebook(GBM_ANCHOR_DOC, gbm_code))

NL = chr(10)


def cells_from_py(src):
    """`# %%` / `# %% [markdown]` blocks -> notebook cells (markdown lines lose their leading '# ')"""
    cells = []
    for block in src.split('# %%')[1:]:
        head, _, body = block.partition(NL)
        if '[markdown]' in head:
            text = NL.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip().splitlines())
            cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))
        else:
            code = ((head.strip() and f'# {head.strip()}' + NL) or '') + body.strip() + NL
            cells.append(dict(cell_type='code', metadata={}, source=code.splitlines(keepends=True),
                              execution_count=None, outputs=[]))
    return cells


final_nb = dict(nbformat=4, nbformat_minor=4,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python')),
                cells=cells_from_py(open('final/make_submission.py', encoding='utf-8').read()))
with open('final/make_submission.ipynb', 'w', encoding='utf-8') as f:
    json.dump(final_nb, f, indent=1, ensure_ascii=False)
print('wrote final/make_submission.ipynb')
