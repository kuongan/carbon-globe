"""Cell-specific blend weights for flat6 (INSIGHTS 4.55): one weight vector per scenario, or per age group.

flat6 uses one weight vector everywhere. For a split of the rows into two cells (A, B), a hybrid probe for member i
= member i on cell A + flat6 on cell B (dead rows exact) gives, with the member's own LB, its alive error on A and on B
up to ONE shared unknown (how flat6's own error splits between A and B). That unknown adds the same constant to every
entry of each cell's error covariance C_cell = (a_i + a_j - D_ij) / 2, and with weights summing to 1 a constant added
to C does not move the optimum: 6 probes (one per member) give the exact best weights per cell.

usage: python cellweights.py make <split>                    split = scn | age   -> result/cw_<split>_<member>.csv
       python cellweights.py solve <split> <LB flat6> <LB probe 1> ... <LB probe 6>   (members in MEMBERS order)
       -> result/submission_blend_flat6_<split>.csv
"""
import sys
import numpy as np
import ensemble as E
import ensemble_scn as S

BASE = 'flat6'
MEMBERS = ['nnx8_y0inv2', 'gbmc3i', 'lstm_y0inv_xinv', 'sw_s42ep6_y0inv_xinv', 'rfm_none', 's7_xinv']
LB = {'nnx8_y0inv2': 0.061, 'gbmc3i': 0.091, 'lstm_y0inv_xinv': 0.109, 'sw_s42ep6_y0inv_xinv': 0.112,
      'rfm_none': 0.155, 's7_xinv': 0.277}
SC = np.array([E.H_SCALE, E.A_SCALE])
YOUNG = (np.array(E.SEED_AGES) <= 50)[:, None]                 # (15, 1)


def cell_a(split, scn):
    """(15, N) bool: rows of cell A (ssp126 for 'scn', ages <= 50 for 'age')"""
    n = 6171
    if split == 'scn':
        return np.full((15, n), scn == 'ssp126')
    return np.broadcast_to(YOUNG, (15, n)).copy()


def load(run):
    return {s: E.load(run, s)[..., :2].astype(np.float64) for s in E.SCENARIOS}


def alive_sq(x, cellmask):
    """sum over the alive rows of a cell of the per-row metric contribution, both scenarios"""
    return sum(0.5 * np.sum((x[s] / SC)[cellmask[s] & ~S.DEAD[s]] ** 2) for s in E.SCENARIOS)


def make(split):
    base = load(BASE)
    for m in MEMBERS:
        p = load(m)
        out = {s: np.where(cell_a(split, s)[..., None], p[s], base[s]) for s in E.SCENARIOS}
        S.write(out, f'result/cw_{split}_{m}.csv')               # dead rows exact, floor at the mapped zero
        print('wrote', f'result/cw_{split}_{m}.csv')


def solve(split, lb_base, lbs):
    N = E.N_ROWS
    P = {m: load(m) for m in MEMBERS}
    A = {s: cell_a(split, s) for s in E.SCENARIOS}
    B = {s: ~A[s] for s in E.SCENARIOS}
    R = len(MEMBERS)
    DA, DB = np.zeros((R, R)), np.zeros((R, R))
    for i in range(R):
        for j in range(R):
            d = {s: P[MEMBERS[i]][s] - P[MEMBERS[j]][s] for s in E.SCENARIOS}
            DA[i, j], DB[i, j] = alive_sq(d, A), alive_sq(d, B)
    dead = np.array([0.5 * sum(S.dead_err(np.concatenate([P[m][s], np.zeros(P[m][s].shape[:2] + (5,))], -1), s)
                               for s in E.SCENARIOS) * N for m in MEMBERS])    # dead-row error inside each member's LB
    # unknown shared constant set to 0: flat6's alive error on B = its total (dead rows exact) -> a_iA = probe - base_B
    fB = lb_base * N
    aA = np.array(lbs) * N - fB
    aB = np.array([LB[m] for m in MEMBERS]) * N - dead - aA
    w = {}
    for name, a, D in (('A', aA, DA), ('B', aB, DB)):
        C = 0.5 * (a[:, None] + a[None, :] - D)
        v = np.linalg.solve(C, np.ones(R)); w[name] = v / v.sum()
    w6 = np.array([0.5367, 0.4011, 0.3049, 0.2778, -0.2602, -0.2603])          # flat6
    def tot(wA, wB):
        CA, CB = 0.5 * (aA[:, None] + aA[None, :] - DA), 0.5 * (aB[:, None] + aB[None, :] - DB)
        return (wA @ CA @ wA + wB @ CB @ wB) / N
    gain = tot(w6, w6) - tot(w['A'], w['B'])
    print('cell A weights', dict(zip(MEMBERS, np.round(w['A'], 3))))
    print('cell B weights', dict(zip(MEMBERS, np.round(w['B'], 3))))
    print(f'predicted gain over flat6 {gain:.4f} -> LB ~{lb_base - gain:.4f}')
    out = {}
    for s in E.SCENARIOS:
        pa = sum(wi * P[m][s] for wi, m in zip(w['A'], MEMBERS))
        pb = sum(wi * P[m][s] for wi, m in zip(w['B'], MEMBERS))
        out[s] = np.where(A[s][..., None], pa, pb)
    S.write(out, f'result/submission_blend_flat6_{split}.csv')
    print('wrote', f'result/submission_blend_flat6_{split}.csv')


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        make(sys.argv[2])
    else:
        solve(sys.argv[2], float(sys.argv[3]), [float(x) for x in sys.argv[4:10]])
