"""Cross-validation ON the public LB, by site parity (even / odd site index).

Public and private LB are a random split of sites, so the public sites fall about equally into even and odd sites.
Splitting every decision into "fit on one half, score on the other" measures how much our LB tuning overfits.

Notation: SE_h(p) = contribution of the public rows of half h to LB(p), so LB(p) = SE_even(p) + SE_odd(p) exactly.

1. Blend weights / epoch selection.  Hybrid probe of run r: r on even sites, base B on odd sites:
       LB(par_r) = SE_even(r) + SE_odd(B)
   With a = SE_odd(B) unknown: SE_even(r) = LB(par_r) - a, SE_odd(r) = LB(r) - LB(par_r) + a.
   Every SE_even is shifted by -a and every SE_odd by +a. For weights summing to 1, the half covariances
   C_h = 0.5 (SE_h,i + SE_h,j - D_h,ij) shift w'C_h w by the same constant, so
     - differences between runs inside one half are exact,
     - cross-fitted scores  w_even' C_odd w_even + w_odd' C_even w_odd  are exact (the shifts cancel).
   D_h (pairwise squared differences on the public rows of half h) is approximated from all rows of that half,
   the same approximation ensemble.py already makes.
2. Bias factors.  8 probes scale one column by ALPHA_B on one (scenario, half) cell of the base file; each gives that
   cell's optimal factor. Factors fitted on one half are scored on the other.
3. Zero-parameter averages (no overfitting possible): `average`.

usage:
  python cv_parity.py make                       -> result/par_<run>.csv for RUNS_CV
  python cv_parity.py solve                      -> per-half table, winner's curse, cross-fitted blend
  ENSEMBLE_FREE=global,global2_co2_hist python cv_parity.py solve   -> same, for the negative-weight (0.079) blend
  PROBE_SUB=<file> PROBE_TAG=<tag> python cv_parity.py make_bias      -> 8 probes result/<tag>_bcv<k>_*.csv
  PROBE_SUB=<file> PROBE_TAG=<tag> python cv_parity.py solve_bias BASE_LB LB_1 ... LB_8
                                                 -> in-sample vs cross-fitted bias gain, result/<tag>_scn_cv.csv
  python cv_parity.py average run1,run2,... TAG  -> result/submission_avg_<TAG>.csv (+ estimated LB if all scored)
Use the LB scores with all the digits Kaggle shows.
"""
import os, sys
import numpy as np
import pandas as pd
import ensemble as E
import probe_bias as PB
from ensemble_scn import write

S = E.SCENARIOS
BASE_RUN = 'bg_scn'                                       # scored file used on the odd half of every hybrid
RUNS_CV = ['noco2_ep4', 'noco2_ep3', 'noco2_ep6', 's7_ep3', 'v2_scheduled_sampling', 'lstm_xl',
           'global', 'global2_co2_hist']          # the last two carry the negative weights of the LB-0.079 blend
# runs allowed a negative weight in the cross-fitted blend (env ENSEMBLE_FREE, as in ensemble.py)
FREE_RUNS = tuple(r for r in os.environ.get('ENSEMBLE_FREE', '').split(',') if r)
PAR_LB = {r: None for r in RUNS_CV}                       # LB of result/par_<run>.csv (fill in after submitting)
PAR_LB.update(noco2_ep3=0.105, noco2_ep6=0.103, s7_ep3=0.106, v2_scheduled_sampling=0.136, global2_co2_hist=0.110,
              noco2_ep4=0.094, lstm_xl=0.129, **{'global': 0.113})
SNAPSHOT_PREFIX = ('noco2', 's7')                         # candidates for the winner's-curse check
ALPHA_B = 1.2
HALVES = ('even', 'odd')


def preds(r):
    return {s: E.load(r, s) for s in S}


def is_even(n):
    return np.arange(n) % 2 == 0


def halves(p):
    return {h: np.concatenate([p[s][:, is_even(p[s].shape[1]) == (h == 'even')] for s in S], axis=1) for h in HALVES}


def half_D(P):
    H = [halves(p) for p in P]
    tot = sum(P[0][s].shape[1] for s in S)
    return {h: H[0][h].shape[1] / tot * E.msd_matrix([x[h] for x in H]) for h in HALVES}


# ---------------------------------------------------------------------------------------------- 1. blend / selection
def make():
    base = preds(BASE_RUN)
    for r in RUNS_CV:
        p = preds(r)
        out = {s: np.where(is_even(p[s].shape[1])[None, :, None], p[s], base[s]) for s in S}
        write(out, f'result/par_{r}.csv')
        print(f'wrote result/par_{r}.csv')


def solve(par_lb=None, verbose=True):
    par_lb = PAR_LB if par_lb is None else par_lb
    runs = [BASE_RUN] + [r for r, v in par_lb.items() if v is not None]
    assert len(runs) >= 3, 'fill in PAR_LB first'
    lb = np.array([E.RUNS[r][0] for r in runs])
    par = np.array([lb[0]] + [par_lb[r] for r in runs[1:]])
    a = 0.5 * lb[0]                                       # arbitrary: cancels in everything reported
    se = {'even': par - a, 'odd': lb - par + a}
    P = [preds(r) for r in runs]
    D = half_D(P)
    C = {h: E.error_cov(se[h], D[h]) for h in HALVES}

    ref = runs.index('noco2_ep4') if 'noco2_ep4' in runs else 0
    if verbose:
        print(f'per-half score minus {runs[ref]} (x2 = in full-LB units; consistent sign in both halves = real)')
        for i, r in enumerate(runs):
            de, do = 2 * (se['even'][i] - se['even'][ref]), 2 * (se['odd'][i] - se['odd'][ref])
            print(f'  {r:24s} even {de:+.4f}  odd {do:+.4f}  | full LB {lb[i] - lb[ref]:+.4f}')

    cand = [i for i, r in enumerate(runs) if r.startswith(SNAPSHOT_PREFIX)]
    regret = None
    if len(cand) >= 2:
        pick = {h: min(cand, key=lambda i: se[h][i]) for h in HALVES}
        regret = sum(se[o][pick[h]] - min(se[o][i] for i in cand) for h, o in (('even', 'odd'), ('odd', 'even')))
        if verbose:
            print(f'winner\'s curse: best snapshot on even = {runs[pick["even"]]}, on odd = {runs[pick["odd"]]}; '
                  f'picking on one half and scoring on the other loses {regret:+.4f} LB vs picking on the scored half')

    free = tuple(runs.index(r) for r in FREE_RUNS if r in runs)
    w = {h: E.best_simplex_weights(C[h], free)[0] for h in HALVES}
    cv = w['even'] @ C['odd'] @ w['even'] + w['odd'] @ C['even'] @ w['odd']
    ins = w['even'] @ C['even'] @ w['even'] + w['odd'] @ C['odd'] @ w['odd']
    wf, mf = E.best_simplex_weights(C['even'] + C['odd'], free)
    if verbose and free:
        print(f'negative weights allowed for {[runs[i] for i in free]}')
    eq = np.ones(len(runs)) / len(runs)
    me = eq @ (C['even'] + C['odd']) @ eq
    if verbose:
        print(f'blend over {runs}')
        print(f'  weights fit on even {dict(zip(runs, np.round(w["even"], 3)))}')
        print(f'  weights fit on odd  {dict(zip(runs, np.round(w["odd"], 3)))}')
        print(f'  in-sample (each half fit on itself) {ins:.4f} | cross-fitted {cv:.4f} | overfit at half size '
              f'{cv - ins:+.4f} (full-size fit overfits ~half as much)')
        print(f'  full fit (= ensemble.py) {mf:.4f} | equal weights {me:.4f}')
    return dict(runs=runs, w=w, wf=wf, cv=cv, ins=ins, mf=mf, me=me, regret=regret, a=a)


# ---------------------------------------------------------------------------------------------- 2. bias factors
CELLS = [(col, scn, h) for col in ('agb', 'height') for scn in S for h in HALVES]


def cell_mask(sub, scn, h):
    parts = sub['id'].str.split('_', expand=True)
    return ((parts[0] == scn) & ((parts[1].astype(int) % 2 == 0) == (h == 'even'))).to_numpy()


def make_bias():
    sub = pd.read_csv(PB.SUB)
    for k, (col, scn, h) in enumerate(CELLS, 1):
        p = sub.copy(); m = cell_mask(sub, scn, h)
        p.loc[m, col] = p.loc[m, col] * ALPHA_B
        path = f'result/{PB.PTAG}_bcv{k}_{col}_{scn}_{h}.csv'
        p.to_csv(path, index=False)
        print(f'LB_{k}: {path}')


def solve_bias(base, *lbs, verbose=True):
    assert len(lbs) == len(CELLS), f'need {len(CELLS)} probe scores'
    sub = pd.read_csv(PB.SUB)
    spp, spy = {}, {}
    for c, l in zip(CELLS, lbs):
        spp[c] = PB.s_pp(sub, c[0], cell_mask(sub, c[1], c[2]))
        spy[c] = PB.s_py_from_probe(spp[c], l - base, ALPHA_B)
    gain = lambda a, c: 0.5 * spp[c] * ((spy[c] / spp[c] - 1) ** 2 - (a - spy[c] / spp[c]) ** 2)
    out, g_in, g_cv, factors = sub.copy(), 0.0, 0.0, {}
    for col in ('agb', 'height'):
        for scn in S:
            ce, co = (col, scn, 'even'), (col, scn, 'odd')
            ae, ao = spy[ce] / spp[ce], spy[co] / spp[co]
            ap = (spy[ce] + spy[co]) / (spp[ce] + spp[co])
            g_in += gain(ap, ce) + gain(ap, co)
            g_cv += gain(ae, co) + gain(ao, ce)
            factors[(col, scn)] = ap
            m = (sub['id'].str.startswith(scn + '_')).to_numpy()
            out.loc[m, col] = sub.loc[m, col] * ap
            if verbose:
                print(f'{col:6s} {scn}: even x{ae:.4f}  odd x{ao:.4f}  pooled x{ap:.4f}')
    path = f'result/{PB.PTAG}_scn_cv.csv'
    out.to_csv(path, index=False)
    if verbose:
        print(f'bias gain: in-sample {g_in:.5f} | cross-fitted {g_cv:.5f} -> expected LB {base - g_in:.5f} '
              f'(pessimistic {base - g_cv:.5f}) | wrote {path}')
    return dict(factors=factors, g_in=g_in, g_cv=g_cv)


# ---------------------------------------------------------------------------------------------- 3. averages
def average(runs_csv, tag):
    rs = [r for r in runs_csv.split(',') if r]
    P = [preds(r) for r in rs]
    out = {s: np.mean([p[s] for p in P], axis=0) for s in S}
    if all(r in E.RUNS for r in rs):
        C = E.error_cov([E.RUNS[r][0] for r in rs], E.msd_matrix([np.concatenate([p[s] for s in S], 1) for p in P]))
        eq = np.ones(len(rs)) / len(rs)
        print(f'estimated LB of the equal-weight average: {eq @ C @ eq:.4f}')
    else:
        print('not every member is scored -> no estimate')
    path = f'result/submission_avg_{tag}.csv'
    write(out, path)
    print('wrote', path)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'make':
        make()
    elif cmd == 'solve':
        solve()
    elif cmd == 'make_bias':
        make_bias()
    elif cmd == 'solve_bias':
        solve_bias(*map(float, sys.argv[2:]))
    elif cmd == 'average':
        average(sys.argv[2], sys.argv[3])
