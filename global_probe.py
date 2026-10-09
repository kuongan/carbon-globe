"""Inference-only experiments with the grid-trained LSTM-XL (global_lstmxl.weights.h5, LB 0.134 / co2_hist 0.122).

Paste into one Kaggle notebook cell (GPU T4 x1 is enough; no repo clone). Inputs: the competition data, the
`zhihaow/carbonglobe` dataset, and a dataset with `global_lstmxl.weights.h5` (optional: `global_lstmxl_preds.npz`
from the same run, for the reproduction check).

Parts (switch them with the flags below):
  A. SCREEN (no submission): on a subsample of test sites, remove the scenario's mean climate shift from ONE feature group
     at a time ("dsh_<group>") and measure how far the year-40 prediction moves and in which direction. Groups that
     move it a lot are the model's levers on the test set, like the CO2 block was. The table also shows the direction
     the LB bias probes ask for (ssp585 agb up, ssp126 height down).
  B. Y0 (no submission): on held-out-style grid sites with known truth, scale the initial state y0 by the test/grid
     ratios and see how much the year-40 prediction remembers y0. Small = the test_y0 shift barely matters.
  C. FULL: rollouts on all test sites for the variants in FULL_VARIANTS -> submission_global_<variant>.csv and
     global_probe_preds.npz (blendable with ensemble.py). Each variant is one LB submission.

Variant grammar: parts joined by '+', applied in order to the raw test forcing of year t:
  co2_hist   CO2 block (8-31) -> grid-mean 1980-2020 series of year t (= the LB-0.122 submission)
  co2_end    CO2 block -> grid-mean 2020 (year 40) values, every year
  co2_start  CO2 block -> grid-mean 1980 (year 1) values, every year
  co2_cont   CO2 block -> grid-mean 2020 values + the test's own increase since its year 1
  co2_up     CO2 block -> co2_hist + (co2_hist - test CO2), i.e. twice as far from the test values (= co2x2)
  co2x<f>    CO2 block -> test + f * (history - test); co2x1 = co2_hist. The LB optimum along global -> global_co2_hist
             was f ~ 1.9 in output space, and the blend with a negative weight on `global` (LB 0.091) relies on it
  f7_fix     feature 7: seasonal anomaly shrunk to the grid's amplitude (test amplitude is ~20x the grid's)
  dsh_<g>    feature group g shifted by -(test mean - grid mean), per month (see GROUPS)
  y0x<f>     initial state y0: every variable times (grid year-0 mean / test y0 mean)^f, per seed age
  y0fx<f>    same for the flux variables only (lai, gpp, npp, rh: test ~1.1-1.3x the grid)
  y0bx<f>    same for the stock variables only (height, agb, soil: test ~0.7-0.85x the grid)
             (ratios with a mean below 1% of Y_MEAN are left at 1, all ratios clipped to [0.25, 4])
"""
import gc, glob, os, re, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.auto import tqdm

ROOT_IN = '/kaggle/input'
COMP = f'{ROOT_IN}/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public'
BASE = f'{ROOT_IN}/datasets/zhihaow/carbonglobe'
OUT = '/kaggle/working'

# Default target: the best single model, no-CO2 seed 42 epoch 4 (LB 0.091). Attach a dataset with its snapshot
# `global_noco2_ckpt/snap_ep4.npz` (result/globel-train/third-run/global_noco2_ckpt/snap_ep4.npz).
CO2_CONST = True                   # weights trained with CO2_MODE='const': the CO2 block is 0 after standardization
WEIGHTS_GLOB = 'snap_ep4.npz'      # file name searched under /kaggle/input when WEIGHTS_FILE is empty
                                   # ('global_lstmxl.weights.h5' for grid run 1/2, CO2_CONST=False)
Z_CLIP = 10.0                      # standardized inputs clipped like in training (None for grid run 1)
SCREEN, SCREEN_SITES = False, 1500    # done for no-CO2 ep4 (INSIGHTS 4.13): only v2 matters
Y0_TEST, Y0_SITES = False, 1000      # done: year-40 keeps ~60% of a height / ~55% of an AGB shift in y0
# 'none' reproduces the scored file (LB 0.091) and costs no slot; the others are one LB submission each
FULL_VARIANTS = (['none', 'dsh_v2', 'y0fx1', 'y0bx1'] if CO2_CONST else
                 ['co2x2', 'co2x3', 'co2_cont', 'co2_end', 'co2_hist+f7_fix', 'co2_hist'])
# rollout state ceiling of carbonglobe-global.ipynb (2 x grid max, printed as 'state ceiling (physical)');
# needed to reproduce runs 2-4 exactly; grid run 1 had none
STATE_CEIL_PHYS = [70.0, 72.322, 213.836, 23.221, 14.709, 7.307, 14.592] if CO2_CONST else None
WEIGHTS_FILE = ''                  # optional explicit path; default: the first WEIGHTS_GLOB match under /kaggle/input
                                   # (attach only ONE snap_ep4.npz: seed 42's, the LB-0.091 model)
GRID_STRIDE = 20                   # grid reference statistics from every 20th site (~2.7k sites, ~0.7 GB read)
BS = 8192
SEED = 0

CO2 = list(range(8, 32))
GROUPS = {**{f'f{i}': [i] for i in range(8)},
          'co2': CO2,
          'v1': list(range(33, 129, 4)), 'v2': list(range(34, 129, 4)),
          'v3': list(range(35, 129, 4)), 'v4': list(range(36, 129, 4)),
          'static': list(range(129, 136))}
TARGET = {'ssp126': dict(h=-4.7, agb=-0.4), 'ssp585': dict(h=-0.3, agb=+8.5)}   # % change wanted by the LB probes

_w = [WEIGHTS_FILE] if WEIGHTS_FILE else sorted(glob.glob(f'{ROOT_IN}/**/{WEIGHTS_GLOB}', recursive=True))
assert _w, f'attach a dataset that contains {WEIGHTS_GLOB}'
if len(_w) > 1:
    print('WARNING: several weight files found, using the first; set WEIGHTS_FILE to choose:', _w)
WEIGHTS = _w[0]

# ------------------------------------------------------------------ constants (same as the notebooks)
N_CONS, N_OUT, N_YEAR, N_T, N_AGE_FEA = 136, 7, 40, 12, 15
N_FEA = N_CONS + N_OUT + N_AGE_FEA
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCENARIOS = ['ssp126', 'ssp585']
N_AGES = len(SEED_AGES)

def normData(inp, mean, std):
    return (inp - mean) / (std + 1e-10)

def invNormData(inp, mean, std):
    return inp * (std + 1e-10) + mean

stats = np.load(f'{BASE}/data_stats/data_stats.npz')
X_MEAN, X_STD = stats['x_mean'].astype(np.float32), stats['x_std'].astype(np.float32)
Y_MEAN, Y_STD = stats['y_mean'].astype(np.float32), stats['y_std'].astype(np.float32)
agetri = np.load(f'{BASE}/data_stats/age_triplet.npz')
TRI_AGE, TRI_MEAN, TRI_SLOPE = (agetri[k].astype(np.float32) for k in ('age', 'age_mean', 'age_slope'))
AGE_MEAN_N, AGE_SLOPE_N = normData(TRI_MEAN, Y_MEAN, Y_STD), normData(TRI_SLOPE, Y_MEAN, Y_STD)
AGE_FEAT = np.stack([np.concatenate([[TRI_AGE[i] / 500.0], AGE_MEAN_N[i], AGE_SLOPE_N[i]])
                     for i in range(N_AGES)]).astype(np.float32)
FLOOR_N = normData(np.zeros(N_OUT, np.float32), Y_MEAN, Y_STD).astype(np.float32)
CEIL_N = (None if STATE_CEIL_PHYS is None else
          normData(np.array(STATE_CEIL_PHYS, np.float32), Y_MEAN, Y_STD).astype(np.float32))

# ------------------------------------------------------------------ LSTM-XL backbone (identical to carbonglobe-global.ipynb)
class GatedBranch(layers.Layer):
    def __init__(self, idx, units, mlp_units, dropout, **kw):
        super().__init__(name=f'branch{idx}', **kw)
        self.idx = idx
        self.lstm_data = layers.LSTM(units, return_sequences=True)
        self.lstm_gate = layers.LSTM(units, return_sequences=True)
        self.gate      = layers.Dense(units, activation='sigmoid')
        self.lstm_out  = layers.LSTM(units, return_sequences=True)
        self.norm      = layers.LayerNormalization(epsilon=1e-5)
        self.pool      = layers.Dense(1)
        self.mlp1      = layers.Dense(mlp_units, activation='gelu')
        self.mlp2      = layers.Dense(128, activation='gelu')
        self.drop      = layers.Dropout(dropout)
        self.head      = layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
    def call(self, h, prev, age, training=False):
        i = self.idx; T = tf.shape(h)[1]
        age_i = tf.concat([age[:, :1], age[:, i+1:i+2], age[:, i+8:i+9]], axis=-1)
        loc = tf.concat([prev[:, i:i+1], age_i], axis=-1)
        z = tf.concat([h, tf.repeat(loc[:, None, :], T, axis=1)], axis=-1)
        s = self.lstm_data(z) * self.gate(self.lstm_gate(z))
        s = self.norm(self.lstm_out(s))
        w = tf.nn.softmax(self.pool(s), axis=1)
        feat = tf.concat([tf.reduce_sum(w * s, axis=1), s[:, -1]], axis=-1)
        feat = self.mlp2(self.drop(self.mlp1(feat), training=training))
        feat = tf.concat([feat, prev, age_i], axis=-1)
        return prev[:, i:i+1] + self.head(feat)

class LSTMXLBackbone(keras.Model):
    def __init__(self, n_out=N_OUT, n_cons=N_CONS, n_age_fea=N_AGE_FEA, d_model=320, n_shared=3,
                 branch_units=320, mlp_units=512, dropout=0.1, return_delta=False, **kw):
        super().__init__(**kw)
        self.n_out, self.n_cons, self.n_age_fea, self.return_delta = n_out, n_cons, n_age_fea, return_delta
        self.ctx_proj = layers.Dense(128, activation='gelu')
        self.in_proj  = layers.Dense(d_model)
        self.in_norm  = layers.LayerNormalization(epsilon=1e-5)
        self.enc_lstms = [layers.Bidirectional(layers.LSTM(d_model // 2, return_sequences=True)) for _ in range(n_shared)]
        self.enc_norms = [layers.LayerNormalization(epsilon=1e-5) for _ in range(n_shared)]
        self.enc_drop  = layers.Dropout(dropout)
        self.branches  = [GatedBranch(i, branch_units, mlp_units, dropout) for i in range(n_out)]
    def call(self, inputs, training=False):
        nc, no, na = self.n_cons, self.n_out, self.n_age_fea
        env = inputs[:, :, :nc]; prev = inputs[:, -1, nc:nc+no]; age = inputs[:, -1, nc+no:nc+no+na]
        T = tf.shape(env)[1]
        ctx = self.ctx_proj(tf.concat([prev, age], axis=-1))
        h = tf.concat([env, tf.repeat(ctx[:, None, :], T, axis=1)], axis=-1)
        h = self.in_norm(self.in_proj(h))
        for lstm, norm in zip(self.enc_lstms, self.enc_norms):
            h = norm(h + self.enc_drop(lstm(h), training=training))
        out = tf.concat([b(h, prev, age, training=training) for b in self.branches], axis=-1)
        return (out - prev) if self.return_delta else out

MODEL = LSTMXLBackbone()
_ = MODEL(tf.zeros((2, N_T, N_FEA), tf.float32), training=False)
if WEIGHTS.endswith('.npz'):                       # snapshot / best.npz of carbonglobe-global.ipynb: b0..bN
    _z = np.load(WEIGHTS)
    _tv = MODEL.trainable_variables
    assert len([k for k in _z.files if k.startswith('b')]) == len(_tv), 'snapshot does not match the model'
    for _k, _v in enumerate(_tv):
        _v.assign(_z[f'b{_k}'])
else:
    MODEL.load_weights(WEIGHTS)
assert sum(float(np.abs(b.head.kernel.numpy()).sum()) for b in MODEL.branches) > 0, 'weights did not load'
print(f'loaded {WEIGHTS} | {MODEL.count_params() / 1e6:.2f}M params')

@tf.function(reduce_retracing=True)
def infer(x):
    return MODEL(x, training=False)

def rollout(env_at, n_sites, y0_raw, desc=None):
    """40-year rollout, all seed ages; env_at(t) -> standardized (n_sites,12,136); y0_raw (15,N,7) -> (15,N,7)."""
    N, A = n_sites, y0_raw.shape[0]
    prev = normData(y0_raw, Y_MEAN, Y_STD).reshape(A * N, N_OUT).astype(np.float32)
    site_idx = np.tile(np.arange(N), A)
    af_rows = AGE_FEAT[np.repeat(np.arange(A), N)]
    for t in tqdm(range(N_YEAR), desc=desc, leave=False, disable=desc is None):
        env_t = env_at(t)
        nxt = np.empty_like(prev)
        for s in range(0, A * N, BS):
            e = min(s + BS, A * N)
            xb = np.concatenate([env_t[site_idx[s:e]], np.repeat(prev[s:e, None, :], N_T, axis=1),
                                 np.repeat(af_rows[s:e, None, :], N_T, axis=1)], axis=-1)
            p = np.maximum(infer(tf.constant(xb, tf.float32)).numpy(), FLOOR_N)
            nxt[s:e] = p if CEIL_N is None else np.minimum(p, CEIL_N)
        prev = nxt
    return np.maximum(invNormData(prev, Y_MEAN, Y_STD).reshape(A, N, N_OUT), 0.0)

def zscore(e):
    """Standardize like in training: clip to +/- Z_CLIP, CO2 block at its mean (0) for no-CO2 weights."""
    z = normData(e, X_MEAN, X_STD)
    if Z_CLIP is not None:
        z = np.clip(z, -Z_CLIP, Z_CLIP)
    if CO2_CONST:
        z[..., CO2] = 0.0
    return z.astype(np.float32)

def msd(a, b):
    return 0.5 * (np.mean(((a[..., 0] - b[..., 0]) / Y_MEAN[0]) ** 2) + np.mean(((a[..., 1] - b[..., 1]) / Y_MEAN[1]) ** 2))

# ------------------------------------------------------------------ grid reference statistics (subsample)
t0 = time.time()
gx = np.load(f'{BASE}/data_global/glob_X_fea.npy', mmap_mode='r')
N_GLOB = gx.shape[0]
GIDX = np.arange(0, N_GLOB, GRID_STRIDE)
g_sum = np.zeros((N_YEAR, N_T, N_CONS)); f7_amp = []
for i in range(0, len(GIDX), 256):
    c = np.asarray(gx[GIDX[i:i + 256]], np.float64)
    g_sum += c.sum(0)
    f7_amp.append(c[..., 7].mean(1).std(1))               # amplitude of each site's mean seasonal cycle
G_YM = g_sum / len(GIDX)                                  # (40, 12, 136) grid mean per year and month
G_M = G_YM.mean(0)                                        # (12, 136) grid monthly mean
G_F7AMP = float(np.median(np.concatenate(f7_amp)))
CO2_HIST = G_YM[..., CO2].astype(np.float32)              # (40, 12, 24)
print(f'grid reference from {len(GIDX)} sites in {time.time() - t0:.0f}s | CO2 block 1980 {CO2_HIST[0].mean():.4g}'
      f' -> 2020 {CO2_HIST[-1].mean():.4g} | feature-7 seasonal amplitude {G_F7AMP:.4g}')

TEST = {}
for scn in SCENARIOS:
    d = np.load(f'{COMP}/test_{scn}.npz')
    tx, ty0 = d['test_x'].astype(np.float32), d['test_y0'].astype(np.float32)
    amp = float(np.median(tx[..., 7].astype(np.float64).mean(1).std(1)))
    TEST[scn] = dict(x=tx, y0=np.transpose(ty0, (1, 0, 2)), delta=tx.astype(np.float64).mean(axis=(0, 1)) - G_M,
                     f7_ratio=amp / (G_F7AMP + 1e-12))
    print(f'{scn}: CO2 block test - grid = {(TEST[scn]["delta"][:, CO2].mean()) / X_STD[CO2].mean():+.2f} X_STD | '
          f'feature-7 amplitude ratio test/grid {TEST[scn]["f7_ratio"]:.2f}')
    del d; gc.collect()

# ---- initial-state variants (y0x / y0fx / y0bx) ----
Y0_FLUX, Y0_STOCK = [3, 4, 5, 6], [0, 1, 2]
NEED_Y0 = any(p.startswith('y0') for v in FULL_VARIANTS for p in v.split('+'))
if NEED_Y0:
    G0 = np.stack([np.asarray(np.load(f'{BASE}/data_global/glob_Y{a:03d}.npy', mmap_mode='r')[GIDX, 0, 11, :],
                              np.float64).mean(axis=0) for a in SEED_AGES])          # (15 ages, 7) grid year-0 mean
    Y0_RATIO = {}
    for scn in SCENARIOS:
        T0 = TEST[scn]['y0'].astype(np.float64).mean(axis=1)                         # (15, 7)
        eps = 0.01 * Y_MEAN[None, :]
        r = np.where((G0 > eps) & (T0 > eps), G0 / np.maximum(T0, 1e-12), 1.0)
        Y0_RATIO[scn] = np.clip(r, 0.25, 4.0)
        print(f'{scn}: grid/test y0 ratio, median over ages [h,agb,soil,lai,gpp,npp,rh]:',
              np.round(np.median(Y0_RATIO[scn], axis=0), 3))

def y0_fn(scn, variant, sites=None):
    y0 = TEST[scn]['y0'] if sites is None else TEST[scn]['y0'][:, sites]
    parts = [p for p in variant.split('+') if p.startswith('y0')]
    if not parts:
        return y0
    y0 = y0.astype(np.float32).copy()
    for p in parts:
        m = re.fullmatch(r'y0([fb]?)x([0-9.]+)', p)
        assert m, f'unknown y0 variant {p!r}'
        idx = {'': list(range(N_OUT)), 'f': Y0_FLUX, 'b': Y0_STOCK}[m.group(1)]
        f = float(m.group(2))
        for v in idx:
            y0[:, :, v] *= (Y0_RATIO[scn][:, v] ** f).astype(np.float32)[:, None]
    return y0

def env_fn(scn, variant, sites=None):
    T = TEST[scn]; tx = T['x'] if sites is None else T['x'][sites]
    parts = [] if variant == 'none' else variant.split('+')
    def env_at(t):
        e = tx[:, t].astype(np.float32).copy()
        for p in parts:
            if p == 'co2_hist':
                e[..., CO2] = CO2_HIST[t]
            elif p == 'co2_end':
                e[..., CO2] = CO2_HIST[-1]
            elif p == 'co2_start':
                e[..., CO2] = CO2_HIST[0]
            elif p == 'co2_cont':
                e[..., CO2] = CO2_HIST[-1] + (tx[:, t][..., CO2] - tx[:, 0][..., CO2])
            elif p == 'co2_up':
                e[..., CO2] = 2 * CO2_HIST[t] - tx[:, t][..., CO2]
            elif p.startswith('co2x'):                       # test + f * (history - test); co2x1 = co2_hist
                f = float(p[4:])
                e[..., CO2] = tx[:, t][..., CO2] + f * (CO2_HIST[t] - tx[:, t][..., CO2])
            elif p == 'f7_fix':
                m = e[..., 7].mean(axis=1, keepdims=True)
                e[..., 7] = m + (e[..., 7] - m) / T['f7_ratio']
            elif p.startswith('dsh_'):
                g = GROUPS[p[4:]]
                e[..., g] -= T['delta'][:, g].astype(np.float32)
            elif p.startswith('y0'):
                continue                                     # handled by y0_fn
            else:
                raise ValueError(f'unknown variant part {p!r}')
        return zscore(e)
    return env_at

def change_row(name, scn, p, ref):
    dh = 100 * (p[..., 0].mean() / ref[..., 0].mean() - 1)
    da = 100 * (p[..., 1].mean() / ref[..., 1].mean() - 1)
    tg = TARGET[scn]
    toward = (dh * tg['h'] + da * tg['agb']) / (np.hypot(tg['h'], tg['agb']) + 1e-12)   # % moved along the wanted direction
    return dict(variant=name, scn=scn, msd_vs_none=msd(p, ref), d_height_pct=dh, d_agb_pct=da, toward_target_pct=toward)

# ------------------------------------------------------------------ A. screening
if SCREEN:
    rng = np.random.default_rng(SEED)
    rows = []
    for scn in SCENARIOS:
        sites = np.sort(rng.choice(TEST[scn]['x'].shape[0], SCREEN_SITES, replace=False))
        y0 = TEST[scn]['y0'][:, sites]
        ref = rollout(env_fn(scn, 'none', sites), len(sites), y0, desc=f'{scn} none')
        co2_names = ['co2x0.5', 'co2_hist', 'co2x1.5', 'co2x2', 'co2x3', 'co2_cont', 'co2_end', 'co2_start']
        names = (([] if CO2_CONST else co2_names) + ['f7_fix']
                 + [f'dsh_{g}' for g in GROUPS if not (CO2_CONST and g == 'co2')])
        for v in tqdm(names, desc=f'screen {scn}'):
            rows.append(change_row(v, scn, rollout(env_fn(scn, v, sites), len(sites), y0), ref))
    scr = pd.DataFrame(rows)
    scr.to_csv(f'{OUT}/global_probe_screen.csv', index=False)
    pd.set_option('display.width', 200)
    print('\n== A. screening (co2_hist row = the change that gained 0.012 on the LB)')
    print(scr.pivot(index='variant', columns='scn', values=['msd_vs_none', 'd_agb_pct', 'd_height_pct', 'toward_target_pct'])
          .round(4).sort_values(('msd_vs_none', 'ssp126'), ascending=False).to_string())

# ------------------------------------------------------------------ B. memory of the initial state
if Y0_TEST:
    rng = np.random.default_rng(SEED + 1)
    gs = np.sort(rng.choice(N_GLOB, Y0_SITES, replace=False))
    gxs = np.asarray(gx[gs], np.float32)
    yg = np.stack([np.load(f'{BASE}/data_global/glob_Y{a:03d}.npy', mmap_mode='r')[gs][:, :, 11, :] for a in SEED_AGES])
    y0g, y40g = yg[:, :, 0], yg[:, :, 40]                                     # (15, n, 7)
    ty0 = np.concatenate([TEST[s]['y0'] for s in SCENARIOS], axis=1)
    ratio = ty0.mean(1, keepdims=True) / (y0g.mean(1, keepdims=True) + 1e-12)  # per age and variable
    env_g = lambda t: zscore(gxs[:, t])
    p_true = rollout(env_g, len(gs), y0g, desc='y0 true')
    p_shift = rollout(env_g, len(gs), y0g * ratio, desc='y0 x test ratio')
    sc = lambda p: 0.5 * (np.mean(((p[..., 0] - y40g[..., 0]) / Y_MEAN[0]) ** 2) + np.mean(((p[..., 1] - y40g[..., 1]) / Y_MEAN[1]) ** 2))
    print('\n== B. y0 memory on', len(gs), 'grid sites (truth known for the true y0 only)')
    print(f'  metric with true y0 {sc(p_true):.5f} | y0 scaled to test ratios: prediction moves msd {msd(p_shift, p_true):.5f}')
    for v, name in ((0, 'height'), (1, 'agb')):
        moved = (p_shift[..., v].mean() / p_true[..., v].mean() - 1) * 100
        put_in = (ratio[..., v] * y0g[..., v]).mean() / y0g[..., v].mean() * 100 - 100
        print(f'  {name}: y0 mean changed {put_in:+.1f}% -> year-40 prediction mean changed {moved:+.1f}%')
    del gxs, yg; gc.collect()

# ------------------------------------------------------------------ C. full-size submissions
if FULL_VARIANTS:
    sample_sub = pd.read_csv(f'{COMP}/sample_submission.csv')
    preds = {}
    for v in FULL_VARIANTS:
        preds[v] = {scn: rollout(env_fn(scn, v), TEST[scn]['x'].shape[0], y0_fn(scn, v), desc=f'{scn} {v}')
                    for scn in SCENARIOS}
        parts = []
        for scn in SCENARIOS:
            p = preds[v][scn]; N = p.shape[1]
            parts.append(pd.DataFrame({'id': [f'{scn}_{s}_{a}' for a in SEED_AGES for s in range(N)],
                                       'height': (p[:, :, 0] / Y_MEAN[0]).reshape(-1),
                                       'agb': (p[:, :, 1] / Y_MEAN[1]).reshape(-1)}))
        sub = sample_sub[['id']].join(pd.concat(parts).set_index('id'), on='id')
        assert len(sub) == len(sample_sub) and not sub[['height', 'agb']].isna().any().any()
        path = f'{OUT}/submission_global_{v.replace("+", "_")}.csv'
        sub.to_csv(path, index=False)
        print(f'wrote {path} | height {sub.height.mean():.3f} agb {sub.agb.mean():.3f}')
    old = glob.glob(f'{ROOT_IN}/**/global_lstmxl_preds.npz', recursive=True)
    if old and 'co2_hist' in preds:
        z = np.load(old[0])
        for scn in SCENARIOS:
            print(f'reproduction {scn}: msd(co2_hist, original run) = {msd(preds["co2_hist"][scn], z[scn + "_co2_hist"]):.2e} '
                  f'(~1e-5 or less: the grid reference here is a {GRID_STRIDE}-site-stride subsample)')
    ref_np = glob.glob(f'{ROOT_IN}/**/global_noco2_preds.npz', recursive=True)   # optional: run 3 output
    if ref_np and 'none' in preds and CO2_CONST:
        z = np.load(ref_np[0])
        if 'ssp126_ep4_none' in z.files:
            for scn in SCENARIOS:
                print(f'reproduction {scn}: msd(none, run-3 epoch 4) = {msd(preds["none"][scn], z[scn + "_ep4_none"]):.2e} '
                      f'(~1e-6 or less: same weights, same preprocessing)')
    if 'none' in preds:
        for v in preds:
            if v != 'none':
                print(f'{v:12s} msd vs none: ' + ' '.join(f'{scn} {msd(preds[v][scn], preds["none"][scn]):.4f}' for scn in SCENARIOS))
    np.savez_compressed(f'{OUT}/global_probe_preds.npz',
                        **{f'{s}_{v.replace("+", "_")}': p for v, d in preds.items() for s, p in d.items()})
    print('wrote', f'{OUT}/global_probe_preds.npz')
