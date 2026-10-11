"""Dead stands in the test (test_y0 height at its mass point = physical 0). The truth there is known exactly (T2,
confirmed by probe_dead_T2.csv = 0.072): a dead stand stays dead, so year-40 = the mapped zero
(height -0.900947 / 10.90645, agb -0.4014121 / 3.39567). Writes result/dead_rows.npz: ssp126/ssp585 masks (15, N)."""
import numpy as np

B_H, B_A = -0.900947, -0.4014121
H_SCALE, A_SCALE = 10.90645, 3.39567
T_H, T_A = B_H / H_SCALE, B_A / A_SCALE           # truth on dead rows, submission units

if __name__ == '__main__':
    out = {}
    for scn in ('ssp126', 'ssp585'):
        y0 = np.load(f'data/public/test_{scn}.npz')['test_y0']             # (N, 15, 7)
        out[scn] = np.transpose(y0[..., 0] == np.float32(B_H))              # (15, N)
        print(scn, 'dead rows', int(out[scn].sum()), f'({out[scn].mean():.2%})')
    np.savez_compressed('result/dead_rows.npz', **out)
    print('wrote result/dead_rows.npz')
