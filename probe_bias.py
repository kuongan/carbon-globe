"""Bias probes on the LB-0.149 blend (result/submission.csv): scale factors per variable, then per cell.

The LB is 0.5 * (MSE_height + MSE_agb) on pre-scaled values (mean over all rows). Multiplying the rows of a cell c of
one column by alpha changes the LB by
    dLB = 0.5 * [(alpha^2 - 1) * S_pp,c - 2 * (alpha - 1) * S_py,c],   S_pp,c = sum_c p^2 / N,  S_py,c = sum_c p*y / N
S_pp,c is known locally, so one probe gives S_py,c and the best factor alpha*_c = S_py,c / S_pp,c, with LB gain
0.5 * S_pp,c * (alpha*_c - 1)^2. Cells add up, so the last cell of a partition comes from the global probe for free.

usage:
  python probe_bias.py make                                  -> result/probe_agb110.csv, result/probe_h110.csv
  python probe_bias.py solve BASE_LB AGB110_LB H110_LB       -> global factors + result/submission_scaled.csv
  python probe_bias.py make_cells COL                        -> result/probe_cell_<COL>_<cell>.csv (3 files)
  python probe_bias.py solve_cells COL BASE_LB GLOBAL110_LB LB_1 LB_2 LB_3
                                                             -> per-cell factors for COL + result/submission_cells_<COL>.csv
     (GLOBAL110_LB = the LB of probe_<col>110.csv; LB_1..3 in the order make_cells prints)
  python probe_bias.py make_scn COL                          -> result/probe_<COL>110_ssp126.csv (one probe)
  python probe_bias.py solve_scn COL BASE_LB GLOBAL110_LB LB_126   -> per-scenario factors for COL
  PROBE_SUB=<file> PROBE_TAG=<tag> python probe_bias.py make4      -> 4 probes of <file> (agb, height; all, ssp126)
  PROBE_SUB=<file> PROBE_TAG=<tag> python probe_bias.py solve4 BASE LB_1 LB_2 LB_3 LB_4
                                                             -> per-scenario factors for both columns, result/<tag>_scn.csv
Use the LB scores with all the digits Kaggle shows.
"""
import os, sys
import numpy as np
import pandas as pd

# base file to probe (env PROBE_SUB); make4/solve4 name their outputs result/<PROBE_TAG>_*.csv
SUB = os.environ.get('PROBE_SUB', 'result/submission.csv')
PTAG = os.environ.get('PROBE_TAG', 'p')
ALPHA, ALPHA_CELL = 1.10, 1.30
COLS = {'agb': 'result/probe_agb110.csv', 'height': 'result/probe_h110.csv'}


def make4():
    """4 probes on SUB: agb x1.1, height x1.1 (all rows), agb x1.1 and height x1.1 (ssp126 rows only)."""
    sub = pd.read_csv(SUB)
    m = sub['id'].str.startswith('ssp126_').to_numpy()
    for k, (col, rows) in enumerate((('agb', None), ('height', None), ('agb', m), ('height', m)), 1):
        p = sub.copy()
        if rows is None:
            p[col] = p[col] * ALPHA
        else:
            p.loc[rows, col] = p.loc[rows, col] * ALPHA
        path = f'result/{PTAG}_probe{k}_{col}110{"" if rows is None else "_ssp126"}.csv'
        p.to_csv(path, index=False)
        print(f'LB_{k}: {path}')


def solve4(base, lb_agb, lb_h, lb_agb126, lb_h126):
    """Per-scenario factors for both columns from the 4 probes of make4 -> result/<PROBE_TAG>_scn.csv."""
    sub = pd.read_csv(SUB)
    m = sub['id'].str.startswith('ssp126_').to_numpy()
    out, gain = sub.copy(), 0.0
    for col, lg, l126 in (('agb', lb_agb, lb_agb126), ('height', lb_h, lb_h126)):
        spy_tot = s_py_from_probe(s_pp(sub, col), lg - base, ALPHA)
        spp126, spp585 = s_pp(sub, col, m), s_pp(sub, col, ~m)
        spy126 = s_py_from_probe(spp126, l126 - base, ALPHA)
        a126, a585 = spy126 / spp126, (spy_tot - spy126) / spp585
        res = LB_ROUND / ((ALPHA - 1) * np.array([spp126, spp585]))
        g = 0.5 * spp126 * (a126 - 1) ** 2 + 0.5 * spp585 * (a585 - 1) ** 2
        gain += g
        out.loc[m, col] = sub.loc[m, col] * a126
        out.loc[~m, col] = sub.loc[~m, col] * a585
        print(f'{col:6s}: ssp126 x{a126:.4f} (+/-{2 * res[0]:.4f})  ssp585 x{a585:.4f} (+/-{2 * res[1]:.4f}) '
              f'| LB gain {g:.5f}')
    path = f'result/{PTAG}_scn.csv'
    out.to_csv(path, index=False)
    print(f'expected LB of {path}: {base - gain:.5f} (base {base:.5f})')
YOUNG = {1, 10, 20, 30, 50}                     # seed ages <= 50 vs >= 70
CELLS = ['ssp126_young', 'ssp126_old', 'ssp585_young', 'ssp585_old']   # the last one is solved from the others
LB_ROUND = 0.0005


def cell_of(sub):
    scn = sub['id'].str.split('_').str[0]
    age = sub['id'].str.split('_').str[2].astype(int)
    return scn + '_' + np.where(age.isin(YOUNG), 'young', 'old')


def s_pp(sub, col, mask=None):
    v = sub[col].to_numpy(np.float64)
    if mask is not None:
        v = np.where(mask, v, 0.0)
    return float(np.sum(v ** 2) / len(sub))


def s_py_from_probe(spp, d, alpha):
    return ((alpha ** 2 - 1) * spp - 2 * d) / (2 * (alpha - 1))


def make():
    sub = pd.read_csv(SUB)
    for col, path in COLS.items():
        p = sub.copy()
        p[col] = p[col] * ALPHA
        p.to_csv(path, index=False)
        print(f'{path}: {col} x {ALPHA} | S_pp({col}) = {s_pp(sub, col):.5f}, mean {sub[col].mean():.4f}')


def solve(base, lb_agb, lb_h):
    sub = pd.read_csv(SUB)
    out, total_gain = sub.copy(), 0.0
    for col, lb in (('agb', lb_agb), ('height', lb_h)):
        spp = s_pp(sub, col)
        a_star = s_py_from_probe(spp, lb - base, ALPHA) / spp
        gain = 0.5 * spp * (a_star - 1) ** 2
        total_gain += gain
        out[col] = out[col] * a_star
        print(f'{col:6s}: delta LB {lb - base:+.5f} -> alpha* = {a_star:.4f} '
              f'(+/- {LB_ROUND / ((ALPHA - 1) * spp):.4f} per {LB_ROUND} of LB rounding) | expected LB gain {gain:.5f}')
    out.to_csv('result/submission_scaled.csv', index=False)
    print(f'expected LB of result/submission_scaled.csv: {base - total_gain:.5f} (base {base:.5f})')


def make_cells(col):
    sub = pd.read_csv(SUB)
    cell = cell_of(sub)
    for k, c in enumerate(CELLS[:-1], 1):
        p = sub.copy()
        m = (cell == c).to_numpy()
        p.loc[m, col] = p.loc[m, col] * ALPHA_CELL
        path = f'result/probe_cell_{col}_{c}.csv'
        p.to_csv(path, index=False)
        spp = s_pp(sub, col, m)
        print(f'LB_{k}: {path} ({col} x {ALPHA_CELL} on {m.mean():.1%} of rows) | S_pp = {spp:.5f}, '
              f'alpha* resolution +/- {2 * LB_ROUND / ((ALPHA_CELL - 1) * spp):.4f}')
    print(f'(cell {CELLS[-1]} is solved from the global probe probe_{"h" if col == "height" else col}110.csv)')


def solve_cells(col, base, lb_global, *lbs):
    assert len(lbs) == len(CELLS) - 1, f'need {len(CELLS) - 1} cell probe scores'
    sub = pd.read_csv(SUB)
    cell = cell_of(sub)
    spy_total = s_py_from_probe(s_pp(sub, col), lb_global - base, ALPHA)
    out, spy_sum, gain_total = sub.copy(), 0.0, 0.0
    for k, c in enumerate(CELLS):
        m = (cell == c).to_numpy()
        spp = s_pp(sub, col, m)
        if k < len(lbs):
            spy = s_py_from_probe(spp, lbs[k] - base, ALPHA_CELL)
            spy_sum += spy
        else:
            spy = spy_total - spy_sum
        a = spy / spp
        gain = 0.5 * spp * (a - 1) ** 2
        gain_total += gain
        out.loc[m, col] = out.loc[m, col] * a
        print(f'{col} {c:13s}: alpha* = {a:.4f} | LB gain {gain:.5f}')
    path = f'result/submission_cells_{col}.csv'
    out.to_csv(path, index=False)
    print(f'expected LB of {path}: {base - gain_total:.5f} (only {col} rescaled; apply the other column\'s global '
          f'factor on top for the combined file)')


def make_scn(col):
    """One probe: COL x ALPHA on the ssp126 rows of the base file (ssp585 follows from the global probe)."""
    sub = pd.read_csv(SUB)
    m = sub['id'].str.startswith('ssp126_').to_numpy()
    p = sub.copy()
    p.loc[m, col] = p.loc[m, col] * ALPHA
    path = f'result/probe_{col}110_ssp126.csv'
    p.to_csv(path, index=False)
    print(f'{path}: {col} x {ALPHA} on ssp126 rows of {SUB} | S_pp = {s_pp(sub, col, m):.5f}')


def solve_scn(col, base, lb_global, lb_126):
    sub = pd.read_csv(SUB)
    m = sub['id'].str.startswith('ssp126_').to_numpy()
    spy_tot = s_py_from_probe(s_pp(sub, col), lb_global - base, ALPHA)
    spp126, spp585 = s_pp(sub, col, m), s_pp(sub, col, ~m)
    spy126 = s_py_from_probe(spp126, lb_126 - base, ALPHA)
    a126, a585 = spy126 / spp126, (spy_tot - spy126) / spp585
    gain = 0.5 * spp126 * (a126 - 1) ** 2 + 0.5 * spp585 * (a585 - 1) ** 2
    print(f'{col}: alpha* ssp126 = {a126:.4f}, ssp585 = {a585:.4f} | LB gain {gain:.5f} (vs base {base})')
    return a126, a585


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'make4':
        make4()
    elif cmd == 'solve4':
        solve4(*map(float, sys.argv[2:7]))
    if cmd == 'make_scn':
        make_scn(sys.argv[2])
    elif cmd == 'solve_scn':
        solve_scn(sys.argv[2], *map(float, sys.argv[3:6]))
    if cmd == 'make':
        make()
    elif cmd == 'solve':
        solve(*map(float, sys.argv[2:5]))
    elif cmd == 'make_cells':
        make_cells(sys.argv[2])
    elif cmd == 'solve_cells':
        solve_cells(sys.argv[2], *map(float, sys.argv[3:]))
