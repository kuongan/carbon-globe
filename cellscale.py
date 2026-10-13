"""Scale one (scenario, variable) cell of a scored submission around the mapped zero, and solve the best factor.

For a cell c the LB is quadratic in the factor f (dead rows sit at the mapped zero b and stay there):
    LB(f) = const + 0.5 * q * f^2 - r * f,   q = sum_c (x - b)^2 / N  (known),   r unknown
so the base LB (f = 1) plus one probe at f1 gives the optimum f* = r / q exactly (up to LB rounding).

usage:
  python cellscale.py make  <base.csv> <scn> <var> <f1>               -> result/cs_<base>_<scn>_<var><f1>.csv
  python cellscale.py solve <base.csv> <scn> <var> <f1> <lb_base> <lb_probe>
  python cellscale.py apply <base.csv> <out.csv> <scn>:<var>:<f> [...]  -> several cells at once
"""
import os, sys
import numpy as np
import pandas as pd

B = {'height': -0.900947 / 10.90645, 'agb': -0.4014121 / 3.39567}     # mapped zero, submission units
N_ROWS = 185130


def scaled(sub, scn, var, f):
    out = sub.copy()
    m = out['id'].str.startswith(scn + '_').to_numpy()
    out.loc[m, var] = B[var] + f * (out.loc[m, var] - B[var])
    return out


def q_of(sub, scn, var):
    m = sub['id'].str.startswith(scn + '_').to_numpy()
    return float(np.sum((sub.loc[m, var].to_numpy() - B[var]) ** 2) / N_ROWS)


def main():
    a = sys.argv
    if a[1] == 'make':
        base, scn, var, f1 = a[2], a[3], a[4], float(a[5])
        out = scaled(pd.read_csv(base), scn, var, f1)
        path = f'result/cs_{os.path.splitext(os.path.basename(base))[0]}_{scn}_{var}{f1:g}.csv'
        assert len(out) == N_ROWS and np.isfinite(out[['height', 'agb']].to_numpy()).all()
        out.to_csv(path, index=False)
        print('wrote', path)
    elif a[1] == 'solve':
        base, scn, var, f1, lb0, lb1 = a[2], a[3], a[4], float(a[5]), float(a[6]), float(a[7])
        q = q_of(pd.read_csv(base), scn, var)
        r = (0.5 * q * (f1 ** 2 - 1) - (lb1 - lb0)) / (f1 - 1)
        fs = r / q
        lbs = lb0 + 0.5 * q * (fs ** 2 - 1) - r * (fs - 1)
        err = 0.001 / (abs(f1 - 1) * q)                 # both LBs shown to 3 decimals
        print(f'{scn} {var}: q {q:.4f} | best factor {fs:.4f} (+-{err:.4f} from LB rounding) | predicted LB {lbs:.4f}')
    elif a[1] == 'apply':
        sub = pd.read_csv(a[2])
        for spec in a[4:]:
            scn, var, f = spec.split(':')
            sub = scaled(sub, scn, var, float(f))
        sub.to_csv(a[3], index=False)
        print('wrote', a[3])


if __name__ == '__main__':
    main()
