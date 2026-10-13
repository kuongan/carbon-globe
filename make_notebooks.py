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

write('kaggle_direct_nn.ipynb', notebook(NN_DOC, open('carbonglobe_direct_nn.py', encoding='utf-8').read()))
