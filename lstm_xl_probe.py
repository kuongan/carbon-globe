"""LSTM-XL (LB 0.168) inference probes on the test set: none / co2_hist / clip.

Paste into one Kaggle notebook cell (or `%run lstm_xl_probe.py`). No training, no repo clone needed.
Kaggle settings: Accelerator = GPU T4 (one is enough). Inputs:
  * the competition data
  * the `zhihaow/carbonglobe` dataset
  * a dataset containing `lstm_xl.weights.h5` (the file saved by carbonglobe-lstm-xl.ipynb)
  * optional: `lstm_xl_preds.npz` (or the unzipped `lstm_xl_preds/` folder) to check that `none` reproduces the
    original run

Variants (each writes /kaggle/working/submission_lstmxl_<variant>.csv):
  none     : test forcing as given  -> should reproduce the LB-0.168 submission
  co2_hist : CO2 block (features 8-31) replaced, year by year, by the 1980-2020 mean CO2 series (with its seasonal
             cycle) of the 3,373 sites the model was trained on
  clip     : every feature clipped to the training range of those sites (fixes the SUS floor offset)
"""
import gc, glob, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.auto import tqdm

ROOT_IN  = '/kaggle/input'
COMP     = f'{ROOT_IN}/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public'
BASE     = f'{ROOT_IN}/datasets/zhihaow/carbonglobe'
OUT      = '/kaggle/working'
VARIANTS = ['co2_hist', 'none', 'clip']
CO2_FEATS = list(range(8, 32))
BS = 8192                                   # rollout batch (rows = site x seed age)

_w = glob.glob(f'{ROOT_IN}/**/lstm_xl.weights.h5', recursive=True)
assert _w, 'attach a dataset that contains lstm_xl.weights.h5'
WEIGHTS = _w[0]

# ------------------------------------------------------------------ constants (same as the LSTM-XL notebook)
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
assert list(TRI_AGE.astype(int)) == SEED_AGES
AGE_MEAN_N, AGE_SLOPE_N = normData(TRI_MEAN, Y_MEAN, Y_STD), normData(TRI_SLOPE, Y_MEAN, Y_STD)
AGE_FEAT = np.stack([np.concatenate([[TRI_AGE[i] / 500.0], AGE_MEAN_N[i], AGE_SLOPE_N[i]])
                     for i in range(N_AGES)]).astype(np.float32)
FLOOR_N = normData(np.zeros(N_OUT, np.float32), Y_MEAN, Y_STD).astype(np.float32)

# ------------------------------------------------------------------ history of the model's own training sites
t0 = time.time()
_res = np.load(f'{BASE}/data_global/res_train4_test8.npz')
_xtr = _res['x_train']                                       # (3373, 40, 12, 136), ~0.9 GB
CO2_HIST = _xtr[..., CO2_FEATS].mean(axis=0)                 # (40, 12, 24): year t of 1980-2020, monthly
LO, HI = _xtr.min(axis=(0, 1, 2)), _xtr.max(axis=(0, 1, 2))
del _xtr, _res; gc.collect()
print(f'training-site history loaded in {time.time() - t0:.0f}s | CO2 block year 1 {CO2_HIST[0].mean():.4g} '
      f'-> year 40 {CO2_HIST[-1].mean():.4g} (raw units)')

# ------------------------------------------------------------------ LSTM-XL (identical to carbonglobe-lstm-xl.ipynb)
class GatedBranch(layers.Layer):
    """Per-target branch: gated LSTM stack -> attention pooling -> MLP -> residual delta."""
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
        i = self.idx
        T = tf.shape(h)[1]
        age_i = tf.concat([age[:, :1], age[:, i + 1:i + 2], age[:, i + 8:i + 9]], axis=-1)
        loc = tf.concat([prev[:, i:i + 1], age_i], axis=-1)
        z = tf.concat([h, tf.repeat(loc[:, None, :], T, axis=1)], axis=-1)
        s = self.lstm_data(z) * self.gate(self.lstm_gate(z))
        s = self.norm(self.lstm_out(s))
        w = tf.nn.softmax(self.pool(s), axis=1)
        feat = tf.concat([tf.reduce_sum(w * s, axis=1), s[:, -1]], axis=-1)
        feat = self.mlp2(self.drop(self.mlp1(feat), training=training))
        feat = tf.concat([feat, prev, age_i], axis=-1)
        return prev[:, i:i + 1] + self.head(feat)

class MM_LSTM_Age_XL(keras.Model):
    def __init__(self, n_out=N_OUT, n_cons=N_CONS, n_age_fea=N_AGE_FEA, d_model=320, n_shared=3,
                 branch_units=320, mlp_units=512, dropout=0.1, **kw):
        super().__init__(**kw)
        self.n_out, self.n_cons, self.n_age_fea = n_out, n_cons, n_age_fea
        self.ctx_proj = layers.Dense(128, activation='gelu')
        self.in_proj  = layers.Dense(d_model)
        self.in_norm  = layers.LayerNormalization(epsilon=1e-5)
        self.enc_lstms = [layers.Bidirectional(layers.LSTM(d_model // 2, return_sequences=True))
                          for _ in range(n_shared)]
        self.enc_norms = [layers.LayerNormalization(epsilon=1e-5) for _ in range(n_shared)]
        self.enc_drop  = layers.Dropout(dropout)
        self.target_branches = [GatedBranch(i, branch_units, mlp_units, dropout) for i in range(n_out)]
    def call(self, inputs, training=False):
        nc, no, na = self.n_cons, self.n_out, self.n_age_fea
        env  = inputs[:, :, :nc]
        prev = inputs[:, -1, nc:nc + no]
        age  = inputs[:, -1, nc + no:nc + no + na]
        T = tf.shape(env)[1]
        ctx = self.ctx_proj(tf.concat([prev, age], axis=-1))
        h = tf.concat([env, tf.repeat(ctx[:, None, :], T, axis=1)], axis=-1)
        h = self.in_norm(self.in_proj(h))
        for lstm, norm in zip(self.enc_lstms, self.enc_norms):
            h = norm(h + self.enc_drop(lstm(h), training=training))
        return tf.concat([b(h, prev, age, training=training) for b in self.target_branches], axis=-1)

MODEL = MM_LSTM_Age_XL()
_ = MODEL(tf.zeros((2, N_T, N_FEA), tf.float32), training=False)
MODEL.load_weights(WEIGHTS)
_head = sum(float(np.abs(b.head.kernel.numpy()).sum()) for b in MODEL.target_branches)
assert _head > 0, 'heads are still zero: the weights did not load'
print(f'loaded {WEIGHTS} | {MODEL.count_params() / 1e6:.2f}M params')

@tf.function(reduce_retracing=True)
def infer(x):
    return MODEL(x, training=False)

def rollout(env_at, n_sites, y0_raw, desc=None):
    """40-year rollout for all seed ages at once; env_at(t) -> standardized (n_sites,12,136). Returns (15,N,7)."""
    N, A = n_sites, y0_raw.shape[0]
    prev = normData(y0_raw, Y_MEAN, Y_STD).reshape(A * N, N_OUT).astype(np.float32)
    site_idx = np.tile(np.arange(N), A)
    af_rows = AGE_FEAT[np.repeat(np.arange(A), N)]
    for t in tqdm(range(N_YEAR), desc=desc, leave=False):
        env_t = env_at(t)
        nxt = np.empty_like(prev)
        for s in range(0, A * N, BS):
            e = min(s + BS, A * N)
            xb = np.concatenate([env_t[site_idx[s:e]],
                                 np.repeat(prev[s:e, None, :], N_T, axis=1),
                                 np.repeat(af_rows[s:e, None, :], N_T, axis=1)], axis=-1)
            nxt[s:e] = np.maximum(infer(tf.constant(xb, tf.float32)).numpy(), FLOOR_N)
        prev = nxt
    return np.maximum(invNormData(prev, Y_MEAN, Y_STD).reshape(A, N, N_OUT), 0.0)

def env_fn(tx, variant):
    def env_at(t):
        e = tx[:, t]
        if variant == 'co2_hist':
            e = e.copy(); e[..., CO2_FEATS] = CO2_HIST[t]
        elif variant == 'clip':
            e = np.clip(e, LO, HI)
        return normData(e, X_MEAN, X_STD).astype(np.float32)
    return env_at

def msd(a, b):
    """mean squared difference in competition units"""
    return 0.5 * (np.mean(((a[..., 0] - b[..., 0]) / Y_MEAN[0]) ** 2) + np.mean(((a[..., 1] - b[..., 1]) / Y_MEAN[1]) ** 2))

def old_pred(scn):
    for p in glob.glob(f'{ROOT_IN}/**/lstm_xl_preds.npz', recursive=True):
        return np.load(p)[scn]
    for p in glob.glob(f'{ROOT_IN}/**/lstm_xl_preds/{scn}.npy', recursive=True):
        return np.load(p)
    return None

# ------------------------------------------------------------------ test rollouts
preds = {v: {} for v in VARIANTS}
for scn in SCENARIOS:
    d = np.load(f'{COMP}/test_{scn}.npz')
    tx, ty0 = d['test_x'].astype(np.float32), d['test_y0'].astype(np.float32)
    co2_z = (tx[..., CO2_FEATS].mean() - CO2_HIST.mean()) / X_STD[CO2_FEATS].mean()
    print(f'{scn}: CO2 block, test - training history = {co2_z:+.2f} X_STD | '
          f'clip touches {np.mean((tx < LO) | (tx > HI)):.2%} of values')
    for v in VARIANTS:
        t1 = time.time()
        preds[v][scn] = rollout(env_fn(tx, v), tx.shape[0], np.transpose(ty0, (1, 0, 2)), desc=f'{scn} {v}')
        print(f'  {v:8s} done in {(time.time() - t1) / 60:.1f} min')
    del tx, ty0, d; gc.collect()

# ------------------------------------------------------------------ submissions + diagnostics
sample_sub = pd.read_csv(f'{COMP}/sample_submission.csv')
for v in VARIANTS:
    parts = []
    for scn in SCENARIOS:
        p = preds[v][scn]; N = p.shape[1]
        parts.append(pd.DataFrame({'id': [f'{scn}_{site}_{age}' for age in SEED_AGES for site in range(N)],
                                   'height': (p[:, :, 0] / Y_MEAN[0]).reshape(-1),
                                   'agb':    (p[:, :, 1] / Y_MEAN[1]).reshape(-1)}))
    sub = sample_sub[['id']].join(pd.concat(parts).set_index('id'), on='id')
    assert len(sub) == len(sample_sub) and not sub[['height', 'agb']].isna().any().any()
    path = f'{OUT}/submission_lstmxl_{v}.csv'
    sub.to_csv(path, index=False)
    both = np.concatenate([preds[v][s] for s in SCENARIOS], axis=1)
    d585 = preds[v]['ssp585'] - preds[v]['ssp126']
    print(f'{v:8s}: height {sub.height.mean():.3f} agb {sub.agb.mean():.3f} (scaled means) | '
          f'ssp585-ssp126 mean d_agb {d585[..., 1].mean() / Y_MEAN[1]:+.4f} | -> {path}')
    if v != 'none' and 'none' in preds:
        ref = np.concatenate([preds['none'][s] for s in SCENARIOS], axis=1)
        print(f'          mean squared difference vs none: {msd(both, ref):.5f}')

if 'none' in preds:
    for scn in SCENARIOS:
        o = old_pred(scn)
        if o is not None:
            print(f'reproduction check {scn}: msd(none, original lstm_xl preds) = {msd(preds["none"][scn], o):.2e} '
                  f'(~0 means the weights and pipeline match the LB-0.168 run)')
np.savez_compressed(f'{OUT}/lstmxl_probe_preds.npz', **{f'{s}_{v}': p for v, d in preds.items() for s, p in d.items()})
print('wrote', f'{OUT}/lstmxl_probe_preds.npz')
