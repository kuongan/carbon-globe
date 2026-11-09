# %% [markdown]
# CarbonGlobe: LightGBM on the global grid (CPU session, no GPU, no repo clone)
#
# Two tree models, each a new blend component (INSIGHTS 4.21: a new member must be different, not just another LSTM):
#   DIRECT : (y0, seed age, 40-year climate summary) -> year-40 height / agb. One row per (site, age), no rollout.
#   ROLLOUT: (state_t, stand age, year-t climate) -> state_t+1 (7 targets), rolled out 40 years like the LSTMs.
# bagging_fraction + feature_fraction = many small learners, each on a random subset of rows and features.
# Trees do not extrapolate (constant beyond the training range), unlike the LSTMs: a different error pattern on the
# shifted test climate. CO2 block + constant feature dropped (no-CO2 won: LB 0.091 vs 0.122).
#
# Transform-aware (INSIGHTS 4.22/4.23): the test inputs AND the hidden truth are `s * physical + b` per variable.
# The trees learn physical dynamics on the grid; test inputs are mapped back to physical units with test_maps.npz
# (written by test_transform.py) and the predictions are mapped forward again before writing the CSV.
#   variant 'inv'  : y0 and forcing mapped back (main)
#   variant 'y0inv': only y0 mapped back, forcing as given
#
# Runs locally (data/ folder, outputs in result/gbm/) or on Kaggle (CPU notebook; competition data +
# zhihaow/carbonglobe + the dataset holding test_maps.npz; outputs in /kaggle/working).
# Outputs:
#   submission_gbm_<model>_<variant>.csv  model in {direct, rollout}
#   gbm_preds.npz      keys <scn>_<model>_<variant>, (15, N, 7), PHYSICAL units (re-map locally with other slopes)
#   gbm_summary.json   holdout metrics, best iterations, feature-group importance, timings

# %%
import os, time, gc, json, re
import numpy as np
import pandas as pd
import lightgbm as lgb

import glob

T0 = time.time()
KAGGLE = os.path.isdir('/kaggle/input')
if KAGGLE:
    COMP = '/kaggle/input/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public'
    BASE = '/kaggle/input/datasets/zhihaow/carbonglobe'
    OUT = '/kaggle/working'
    MAPS = sorted(glob.glob('/kaggle/input/**/test_maps.npz', recursive=True))[0]
else:                                                # local copy of the data (repo root as working directory)
    COMP, BASE, OUT = 'data/public', 'data/carbonglobe', 'result/gbm'
    MAPS = 'kaggle_upload/noco2-ep4/test_maps.npz'
os.makedirs(OUT, exist_ok=True)
_mz = np.load(MAPS)
MAP_XS, MAP_XB = _mz['x_s'].astype(np.float32), _mz['x_b'].astype(np.float32)
MAP_YS, MAP_YB = _mz['y_s'].astype(np.float32), _mz['y_b'].astype(np.float32)

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCENARIOS = ['ssp126', 'ssp585']
N_AGES, N_YEAR, N_T, N_CONS, N_OUT = len(SEED_AGES), 40, 12, 136, 7
H_SCALE, A_SCALE = 10.90645, 3.39567                 # submission scale = Y_MEAN[0], Y_MEAN[1]
STATE_NAMES = ['height', 'agb', 'soil', 'lai', 'gpp', 'npp', 'rh']
STOCK = (0, 1, 2)                                    # rollout: stocks predicted as a yearly change, fluxes as a level

# ---- config ----
SEED = int(os.environ.get('GBM_SEED', 42))          # env overrides for local bagging runs (seed = trees, rows,
_models = os.environ.get('GBM_MODELS', 'direct,rollout').split(',')   # features and the holdout blocks)
RUN_DIRECT, RUN_ROLLOUT = 'direct' in _models, 'rollout' in _models
TAG = os.environ.get('GBM_TAG', '')                  # suffix of every output file, e.g. '_s1'
VARIANTS = tuple(os.environ.get('GBM_VARIANTS', 'inv,y0inv').split(','))   # y0inv scored 0.108, inv 0.164
HOLD_FRAC, BLOCK_DEG = 0.08, 10       # holdout = 8% of 10-degree blocks (neighbouring 0.5-degree cells are near-copies)
ROLL_ROWS, ROLL_VAL_ROWS = 2_000_000, 200_000        # one-step (site, age, year) rows sampled for the rollout model
                                                     # (peak RAM ~6 GB; 3M rows on Kaggle's 30 GB is fine too)
STATE_CEIL = 2.0                      # rollout states capped at 2x the grid maximum (as in the LSTM notebook)
LOAD_CHUNK = 4096 if KAGGLE else 1024                # Kaggle CPU sessions have ~30 GB RAM
PARAMS = dict(objective='l2', learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=127,
              num_threads=(os.cpu_count() or 4) if KAGGLE else max(1, (os.cpu_count() or 4) - 2),   # local: 2 cores free
              verbose=-1, seed=SEED)
DIRECT_ROUNDS, ROLL_ROUNDS, ES_ROUNDS = 5000, 3000, 100
# env GBM_BOOSTING=rf (INSIGHTS 4.33): LightGBM's random-forest mode, i.e. bagged deep trees averaged instead of
# boosted, a different tree family from the same library (XGBoost / CatBoost would be more boosting)
if os.environ.get('GBM_BOOSTING', 'gbdt') == 'rf':
    PARAMS.update(boosting='rf', num_leaves=1023, min_data_in_leaf=50, bagging_fraction=0.632, bagging_freq=1,
                  feature_fraction=0.5)
    DIRECT_ROUNDS, ROLL_ROUNDS, ES_ROUNDS = 300, 300, 50

CO2 = list(range(8, 32))
# env GBM_CHRONO=1 (INSIGHTS 4.44): two feature groups the direct model never had (grid holdout 0.0336 -> 0.0270)
#   chrono: the site's own chronosequence = y0 height/agb of all 15 seed ages + y0 interpolated at age a+40
#   risk  : site-year collapse events (>30% agb lost in a year, same year across ages, AUC 0.985 from that year's
#           monthly climate + state) -> event classifier on the grid, summed / timed probabilities over the 40 years
CHRONO = os.environ.get('GBM_CHRONO', '0') == '1'
FIRE40 = os.environ.get('GBM_FIRE40', '0') == '1'    # + every year's event probability (featlab: 0.0270 -> 0.0264)
MAG = os.environ.get('GBM_MAG', '0') == '1'          # + expected one-year agb loss (featlab 0.0270 -> 0.0260, A6)
# GBM_ANCHOR=1: the direct target is y40 - (the site's own y0 at age a+40) instead of y40 - own y0: the level comes
# from the site's chronosequence (in the same space as the answer), the trees only learn the correction
ANCHOR = os.environ.get('GBM_ANCHOR', '0') == '1'
EV_SEEDS, EV_ROUNDS = (6, 10, 14), 600                           # seed ages 100 / 300 / 500 label the events
EV_ROWS = 4_000_000 if KAGGLE else 2_000_000                     # event-classifier rows (Kaggle: more RAM)
KEEP = [j for j in range(N_CONS) if j not in CO2 and j != 32]   # 111 features: no CO2 block, no constant
MON = list(range(7))      # f0-f6 also kept monthly; f7's test seasonal amplitude is ~20x the grid's -> annual mean only
V2 = list(range(34, 129, 4))
rng = np.random.default_rng(SEED)


def fname(j):
    if j < 8:
        return f'f{j}'
    if j < 129:
        return f'v{(j - 33) % 4 + 1}_L{(j - 33) // 4}'
    return f'static{j - 129}'


def group_of(n):
    if n.startswith(('st_', 'y0_')):
        return 'state'
    if n.endswith('age'):
        return 'age'
    m = re.search(r'(f\d|v\d|static)', n)
    return m.group(1) if m else 'other'


def stamp(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def metric(p, t):
    """Competition metric on height & agb, physical units in."""
    return 0.5 * (np.mean(((p[..., 0] - t[..., 0]) / H_SCALE) ** 2) + np.mean(((p[..., 1] - t[..., 1]) / A_SCALE) ** 2))


print('lightgbm', lgb.__version__, '| threads', PARAMS['num_threads'], '| features kept', len(KEEP))

# %% Load the grid: annual means of every feature + monthly f0-f6, annual states (month 11) for the 15 seed ages
D = f'{BASE}/data_global'
gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
N_GLOB = gx.shape[0]
assert gx.shape[1:] == (N_YEAR, N_T, N_CONS), gx.shape
A_all = np.empty((N_GLOB, N_YEAR, N_CONS), np.float32)          # raw units (trees need no standardization)
M_all = np.empty((N_GLOB, N_YEAR, N_T, len(MON)), np.float32)
X_BAD = np.zeros(N_GLOB, bool)
for s in range(0, N_GLOB, LOAD_CHUNK):
    e = min(s + LOAD_CHUNK, N_GLOB)
    c = np.array(gx[s:e], dtype=np.float32)
    X_BAD[s:e] = ~np.isfinite(c).all(axis=(1, 2, 3))
    A_all[s:e] = c.mean(axis=2)
    M_all[s:e] = c[..., MON]
    del c
    if (s // LOAD_CHUNK) % 10 == 0:
        stamp(f'glob_X {e}/{N_GLOB}')
del gx; gc.collect()

y_all = np.empty((N_AGES, N_GLOB, N_YEAR + 1, N_OUT), np.float32)
for ai, age in enumerate(SEED_AGES):
    g = np.load(f'{D}/glob_Y{age:03d}.npy', mmap_mode='r')
    assert g.shape == (N_GLOB, N_YEAR + 1, N_T, N_OUT), g.shape
    for s in range(0, N_GLOB, 8192):
        y_all[ai, s:s + 8192] = g[s:s + 8192, :, 11, :]
    del g
SITE_OK = ~X_BAD & np.isfinite(y_all).all(axis=(0, 2, 3))
OK_IDX = np.flatnonzero(SITE_OK)
PHYS_MAX = np.max([y_all[:, OK_IDX[i:i + 4096]].max(axis=(0, 1, 2)) for i in range(0, len(OK_IDX), 4096)], axis=0)
CEIL = (STATE_CEIL * PHYS_MAX).astype(np.float32)
_gs = np.zeros(N_CONS)
for i in range(0, len(OK_IDX), 4096):
    _gs += A_all[OK_IDX[i:i + 4096]].astype(np.float64).sum(axis=(0, 1))
GRID_MEAN = (_gs / (len(OK_IDX) * N_YEAR)).astype(np.float32)        # grid mean of each feature, raw units
stamp(f'grid loaded: usable sites {len(OK_IDX)}/{N_GLOB} | state ceiling {np.round(CEIL, 2)}')

# ---- holdout: random 10-degree blocks (sanity check + early stopping only; it cannot rank models for the LB) ----
_r, _c = np.nonzero(np.load(f'{BASE}/data_stats/mask2d.npy') > 0)    # grid order = row-major valid cells
assert len(_r) == N_GLOB, 'mask2d does not map 1:1 to the grid'
_cells = int(round(BLOCK_DEG / 0.5))
BLOCK = (_r // _cells) * 100000 + (_c // _cells)
_ub = np.unique(BLOCK[SITE_OK])
_hb = rng.choice(_ub, size=max(1, int(round(HOLD_FRAC * len(_ub)))), replace=False)
IS_HOLD = np.isin(BLOCK, _hb)
TRAIN = np.flatnonzero(SITE_OK & ~IS_HOLD)
HOLD = np.flatnonzero(SITE_OK & IS_HOLD)
hold_y40 = y_all[:, HOLD, N_YEAR].copy()
hold_y0 = y_all[:, HOLD, 0].copy()
stamp(f'train sites {len(TRAIN)} | holdout sites {len(HOLD)} ({len(_hb)}/{len(_ub)} blocks) | '
      f'holdout persistence (y40 = y0) metric {metric(hold_y0, hold_y40):.4f}')

# ---- test: map the inputs back to physical units (test = s * physical + b per variable) ----
_ss = f'{COMP}/sample_submission.csv'
sample_ids = (pd.read_csv(_ss)['id'] if os.path.isfile(_ss) else       # local download has no sample_submission
              pd.Series([f'{scn}_{s}_{a}' for scn in SCENARIOS for s in range(6171) for a in SEED_AGES]))
X_KEEP = np.ones(N_CONS, bool); X_KEEP[CO2] = False; X_KEEP[32] = False
TEST = {}
for scn in SCENARIOS:
    d = np.load(f'{COMP}/test_{scn}.npz')
    tx = d['test_x'].astype(np.float32)                                   # (N, 40, 12, 136)
    y0_t = np.transpose(d['test_y0'].astype(np.float32), (1, 0, 2))     # (15, N, 7), test space
    y0_p = np.maximum((y0_t - MAP_YB) / MAP_YS, 0.0).astype(np.float32)  # physical; dead stands -> exactly 0
    A, M = {}, {}
    # 'none' / 'xinv' (INSIGHTS 4.29): y0 left in the test space (as if physical), forcing as given / mapped back;
    # the twins of y0inv / inv, so the blend can extrapolate along the y0 inversion like the LSTM pair (x1, LB 0.066)
    # 'y0inv2' (INSIGHTS 4.32): y0 pushed past physical by the same step, y0_p + (y0_p - y0_t), dead stands stay dead;
    # both blend pairs (LSTM, GBM) extrapolate in that direction
    assert set(VARIANTS) <= {'inv', 'y0inv', 'none', 'xinv', 'y0inv2'}, VARIANTS
    for v in set(VARIANTS) & {'y0inv', 'none', 'y0inv2'}:                 # forcing as given
        A[v] = tx.mean(axis=2); M[v] = np.ascontiguousarray(tx[..., MON])
    if set(VARIANTS) & {'inv', 'xinv'}:                                   # forcing mapped back, in place (RAM)
        for j in np.flatnonzero(X_KEEP):
            tx[..., j] -= MAP_XB[j]; tx[..., j] /= MAP_XS[j]
        for v in set(VARIANTS) & {'inv', 'xinv'}:
            A[v] = tx.mean(axis=2); M[v] = np.ascontiguousarray(tx[..., MON])
    y0_x = np.where(y0_p[..., :1] > 0, np.maximum(2 * y0_p - y0_t, 0.0), 0.0).astype(np.float32)
    Y0 = {v: (y0_t if v in ('none', 'xinv') else y0_x if v == 'y0inv2' else y0_p) for v in VARIANTS}
    TEST[scn] = dict(A=A, M=M, y0=Y0)
    stamp(f'{scn}: test {tx.shape} | dead (y0 at the mapped zero) {np.mean(y0_p[..., 0] == 0):.4f} | '
          f'non-finite test inputs {int((~np.isfinite(tx)).sum())}')
    del tx, d; gc.collect()


def write_submission(p_by_scn, path):
    """Physical predictions -> the test/truth space (s * p + b; physical 0 -> the mass point) -> /global mean."""
    parts = []
    for scn in SCENARIOS:
        p = np.maximum(p_by_scn[scn], 0.0) * MAP_YS + MAP_YB; N = p.shape[1]
        ids = [f'{scn}_{site}_{age}' for age in SEED_AGES for site in range(N)]
        parts.append(pd.DataFrame({'id': ids, 'height': (p[:, :, 0] / H_SCALE).reshape(-1),
                                   'agb': (p[:, :, 1] / A_SCALE).reshape(-1)}))
    sub = pd.DataFrame({'id': sample_ids}).join(pd.concat(parts).set_index('id'), on='id')
    assert len(sub) == len(sample_ids) and not sub[['height', 'agb']].isna().any().any(), 'missing ids / NaNs'
    sub.to_csv(path, index=False)
    return sub


def importance(m):
    g = pd.Series(m.feature_importance('gain'), index=m.feature_name())
    g = g / max(g.sum(), 1e-12)
    by = g.groupby([group_of(n) for n in g.index]).sum().sort_values(ascending=False)
    return {k: round(float(v), 3) for k, v in by.items()}, g.sort_values(ascending=False).head(10).round(3).to_dict()


PREDS, SUMMARY = {}, dict(params=PARAMS, n_train_sites=int(len(TRAIN)), n_hold_sites=int(len(HOLD)),
                          hold_persistence=float(metric(hold_y0, hold_y40)))


def save_outputs():
    np.savez_compressed(f'{OUT}/gbm_preds{TAG}.npz', **{f'{scn}_{k}': p for k, d in PREDS.items() for scn, p in d.items()})
    with open(f'{OUT}/gbm_summary{TAG}.json', 'w') as f:
        json.dump(SUMMARY, f, indent=1, default=str)
    stamp(f'saved gbm_preds{TAG}.npz ({sorted(PREDS)}) + gbm_summary{TAG}.json')


def report(key):
    p = PREDS[key]
    sc = metric(p['ssp126'], p['ssp585'])
    m = {scn: (float(p[scn][..., 0].mean() / H_SCALE), float(p[scn][..., 1].mean() / A_SCALE)) for scn in SCENARIOS}
    stamp(f'{key}: scaled means (height, agb) {m} | msd ssp126 vs ssp585 {sc:.5f}')
    SUMMARY.setdefault('test', {})[key] = dict(means=m, msd_between_scenarios=float(sc))


# %% DIRECT model: year-40 height / agb from y0 + a 40-year climate summary
DNAMES = ([f'y0_{v}' for v in STATE_NAMES] + ['seed_age'] + [f'mean_{fname(j)}' for j in KEEP]
          + [f'dec{d}_f{i}' for d in range(4) for i in range(8)]
          + [f'clim_f{i}_m{m}' for m in range(N_T) for i in MON])


def direct_clim(A, M):
    """A (n, 40, 136) annual means, M (n, 40, 12, 7) monthly f0-f6 -> (n, F) per-site climate summary."""
    n = A.shape[0]
    return np.concatenate([A[:, :, KEEP].mean(1),                                  # 40-year mean (111)
                           A[:, :, :8].reshape(n, 4, 10, 8).mean(2).reshape(n, 32),  # decade means of f0-f7
                           M.mean(1).reshape(n, N_T * len(MON))], 1).astype(np.float32)   # monthly climatology f0-f6


def direct_X(y0, clim, extra=None):
    """y0 (15, n, 7), clim (n, F), extra (15, n, E) or None -> (15 n, 8 + F [+ E]), rows ordered (age, site)."""
    n = clim.shape[0]
    return np.concatenate([np.concatenate([y0[a], np.full((n, 1), SEED_AGES[a], np.float32), clim]
                                          + ([extra[a]] if extra is not None else []), 1)
                           for a in range(N_AGES)], 0)


AG_F = np.asarray(SEED_AGES, np.float64)


def chrono_feats(y0):
    """y0 (15, n, 7) -> (15, n, 34): all ages' height/agb (30), y0 interpolated at age a+40 (2), its gap to y0 (2)"""
    n = y0.shape[1]
    allages = np.broadcast_to(y0[:, :, :2].transpose(1, 0, 2).reshape(1, n, 30), (N_AGES, n, 30))
    ana = np.empty((N_AGES, n, 2), np.float32)
    for i, a in enumerate(np.minimum(AG_F + 40, 500)):
        j = min(max(int(np.searchsorted(AG_F, a)), 1), N_AGES - 1); w = (a - AG_F[j - 1]) / (AG_F[j] - AG_F[j - 1])
        ana[i] = (1 - w) * y0[j - 1, :, :2] + w * y0[j, :, :2]
    return np.concatenate([allages, ana, ana - y0[:, :, :2]], 2).astype(np.float32)


def ev_X(state, t, A_t, A_p, M_t, M_p):
    """event features: state (b, 7), year t, annual means (b, 136) and monthly f0-f6 (b, 12, 7) of year t and t-1"""
    b = len(state)
    return np.concatenate([M_t.reshape(b, -1), M_p.reshape(b, -1), A_t[:, KEEP], state,
                           np.full((b, 1), t, np.float32)], 1).astype(np.float32)


def risk_feats(clf, y0, A, M):
    """(15, n, 5 [+40]) from the event classifier at the y0 state: sum, max, time-weighted sum, last-decade sum, peak
    year [, every year's probability with GBM_FIRE40=1]"""
    n = y0.shape[1]; out = np.zeros((N_AGES, n, 45 if FIRE40 else 5), np.float32); w = np.arange(N_YEAR) / N_YEAR
    for s0 in range(0, n, 4096):
        sl = slice(s0, min(s0 + 4096, n))
        for a in range(N_AGES):
            P = np.stack([clf.predict(ev_X(y0[a, sl], t, A[sl, t], A[sl, max(t - 1, 0)], M[sl, t], M[sl, max(t - 1, 0)]))
                          for t in range(N_YEAR)], 1)
            out[a, sl, :5] = np.stack([P.sum(1), P.max(1), (P * w).sum(1), P[:, 30:].sum(1), P.argmax(1) / N_YEAR], 1)
            if FIRE40:
                out[a, sl, 5:] = P
    return out


def mag_feats(reg, y0, A, M):
    """(15, n, 4) expected one-year agb loss at the y0 state over the 40 years: sum, max, last decade, survival"""
    n = y0.shape[1]; out = np.zeros((N_AGES, n, 4), np.float32)
    for s0 in range(0, n, 4096):
        sl = slice(s0, min(s0 + 4096, n))
        for a in range(N_AGES):
            L = np.stack([reg.predict(ev_X(y0[a, sl], t, A[sl, t], A[sl, max(t - 1, 0)], M[sl, t], M[sl, max(t - 1, 0)]))
                          for t in range(N_YEAR)], 1)
            out[a, sl] = np.stack([L.sum(1), L.max(1), L[:, 30:].sum(1), np.prod(1 - np.clip(L, 0, 1), 1)], 1)
    return out


if CHRONO:
    DNAMES += ([f'chr_{v}{a}' for a in SEED_AGES for v in ('h', 'a')] + ['chr_ana_h', 'chr_ana_a', 'chr_gap_h',
               'chr_gap_a', 'risk_sum', 'risk_max', 'risk_tw', 'risk_last10', 'risk_peak'])
    if FIRE40:
        DNAMES += [f'risk_y{t}' for t in range(N_YEAR)]
    if MAG:
        DNAMES += ['mag_sum', 'mag_max', 'mag_last10', 'mag_surv']


def base_of(y0):
    """the level the direct model predicts a change from: own y0, or (GBM_ANCHOR) the site's y0 at age a+40"""
    return chrono_feats(y0)[..., 30:32] if ANCHOR else y0[..., :2]


def direct_predict(models, y0, clim, extra=None):
    X = direct_X(y0, clim, extra)
    p = y0.copy()                                     # channels 2-6 are not predicted: left at y0 (not scored)
    dead = (y0[..., 0] == 0) & (y0[..., 1] == 0)      # on the grid a dead stand stays dead for 40 years (100%)
    base = base_of(y0)
    for k, m in models.items():
        p[..., k] = np.where(dead, 0.0, np.clip(base[..., k] + m.predict(X).reshape(N_AGES, -1), 0.0, CEIL[k]))
    return p


if RUN_DIRECT:
    def clim_of(idx):
        return np.concatenate([direct_clim(A_all[idx[i:i + 4096]], M_all[idx[i:i + 4096]])
                               for i in range(0, len(idx), 4096)], 0)
    EXTRA, CLF = {}, None
    if CHRONO:
        s_ = rng.choice(TRAIN, EV_ROWS); a_ = rng.choice(EV_SEEDS, EV_ROWS); t_ = rng.integers(0, N_YEAR, EV_ROWS)
        st_ = y_all[a_, s_, t_]; nx_ = y_all[a_, s_, t_ + 1]
        keep_ = st_[:, 1] > 0
        lab_ = (nx_[:, 1] < 0.7 * st_[:, 1])[keep_]
        loss_ = np.clip(1 - nx_[:, 1] / np.maximum(st_[:, 1], 1e-6), 0, 1)[keep_]
        s_, t_, st_ = s_[keep_], t_[keep_], st_[keep_]
        tp_ = np.maximum(t_ - 1, 0)
        Xe = np.concatenate([M_all[s_, t_].reshape(len(s_), -1), M_all[s_, tp_].reshape(len(s_), -1),
                             A_all[s_, t_][:, KEEP], st_, t_[:, None].astype(np.float32)], 1).astype(np.float32)
        CLF = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                             feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=SEED,
                             num_threads=PARAMS['num_threads']), lgb.Dataset(Xe, lab_.astype(np.float32)), EV_ROUNDS)
        REG = None
        if MAG:
            REG = lgb.train(dict(objective='l2', learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                                 feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=SEED,
                                 num_threads=PARAMS['num_threads']), lgb.Dataset(Xe, loss_.astype(np.float32)), EV_ROUNDS)
        del Xe; gc.collect()
        stamp(f'event classifier: {len(lab_):,} site-years, event rate {lab_.mean():.4f} | magnitude model {MAG}')

        def extra_of(idx):
            parts = []
            for i in range(0, len(idx), 4096):
                j = idx[i:i + 4096]; y0_ = y_all[:, j, 0]
                parts.append(np.concatenate([chrono_feats(y0_), risk_feats(CLF, y0_, A_all[j], M_all[j])]
                                            + ([mag_feats(REG, y0_, A_all[j], M_all[j])] if MAG else []), 2))
            return np.concatenate(parts, 1)
        EXTRA['train'], EXTRA['hold'] = extra_of(TRAIN), extra_of(HOLD)
        stamp('chrono + risk features done (grid)')
    Xtr = direct_X(y_all[:, TRAIN, 0], clim_of(TRAIN), EXTRA.get('train'))
    Xva = direct_X(hold_y0, clim_of(HOLD), EXTRA.get('hold'))
    assert Xtr.shape[1] == len(DNAMES), (Xtr.shape, len(DNAMES))
    dmodels = {}
    for k in (0, 1):
        ytr = (y_all[:, TRAIN, N_YEAR, k] - base_of(y_all[:, TRAIN, 0])[..., k]).reshape(-1)
        yva = (hold_y40[..., k] - base_of(hold_y0)[..., k]).reshape(-1)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=DNAMES, params=PARAMS, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr, params=PARAMS)
        dmodels[k] = lgb.train(PARAMS, dtr, DIRECT_ROUNDS, valid_sets=[dva], valid_names=['hold'],
                               callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(500)])
        gi, top = importance(dmodels[k])
        SUMMARY.setdefault('direct', {})[STATE_NAMES[k]] = dict(best_iter=dmodels[k].best_iteration, groups=gi, top=top)
        stamp(f'direct {STATE_NAMES[k]}: best iter {dmodels[k].best_iteration} | gain by group {gi}')
        del dtr, dva; gc.collect()
    del Xtr, Xva; gc.collect()
    hp = direct_predict(dmodels, hold_y0, clim_of(HOLD), EXTRA.get('hold'))
    SUMMARY['direct']['hold_metric'] = float(metric(hp, hold_y40))
    stamp(f'direct holdout metric {metric(hp, hold_y40):.4f} (persistence {SUMMARY["hold_persistence"]:.4f})')
    for v in VARIANTS:
        key = f'direct_{v}'
        PREDS[key] = {}
        for scn, T in TEST.items():
            ex = (np.concatenate([chrono_feats(T['y0'][v]), risk_feats(CLF, T['y0'][v], T['A'][v], T['M'][v])]
                                 + ([mag_feats(REG, T['y0'][v], T['A'][v], T['M'][v])] if MAG else []), 2)
                  if CHRONO else None)
            PREDS[key][scn] = direct_predict(dmodels, T['y0'][v], direct_clim(T['A'][v], T['M'][v]), ex)
        write_submission(PREDS[key], f'{OUT}/submission_gbm_{key}{TAG}.csv')
        report(key)
    save_outputs()

# %% ROLLOUT model: one step per year, 7 targets, rolled out 40 years from y0
RNAMES = ([f'st_{v}' for v in STATE_NAMES] + ['stand_age'] + [fname(j) for j in KEEP]
          + [f'{fname(i)}_m{m}' for m in range(N_T) for i in MON])


def roll_X(state, age, A_t, M_t):
    """state (n, 7), age (n,), A_t (n, 136), M_t (n, 12, 7) -> (n, F)."""
    return np.concatenate([state, age[:, None], A_t[:, KEEP], M_t.reshape(len(state), -1)], 1).astype(np.float32)


def sample_rows(sites, n_rows):
    s = rng.choice(sites, n_rows); a = rng.integers(0, N_AGES, n_rows); t = rng.integers(0, N_YEAR, n_rows)
    X = np.empty((n_rows, len(RNAMES)), np.float32); Y = np.empty((n_rows, N_OUT), np.float32)
    for i in range(0, n_rows, 500_000):
        j = slice(i, i + 500_000)
        st = y_all[a[j], s[j], t[j]]
        nx = y_all[a[j], s[j], t[j] + 1]
        X[j] = roll_X(st, np.asarray(SEED_AGES, np.float32)[a[j]] + t[j], A_all[s[j], t[j]], M_all[s[j], t[j]])
        Y[j] = nx
        Y[j, list(STOCK)] = nx[:, list(STOCK)] - st[:, list(STOCK)]      # stocks: yearly change
    return X, Y


def rollout(models, y0, A, M):
    """y0 (15, n, 7), A (n, 40, 136), M (n, 40, 12, 7) -> year-40 state (15, n, 7), physical units."""
    n = y0.shape[1]
    st = y0.reshape(N_AGES * n, N_OUT).astype(np.float32).copy()
    age0 = np.repeat(np.asarray(SEED_AGES, np.float32), n)
    si = np.tile(np.arange(n), N_AGES)
    dead = (st[:, 0] == 0) & (st[:, 1] == 0)          # dead at year 0 stays dead (100% on the grid)
    for t in range(N_YEAR):
        X = roll_X(st, age0 + t, A[si, t], M[si, t])
        new = np.empty_like(st)
        for k in range(N_OUT):
            p = models[k].predict(X)
            new[:, k] = st[:, k] + p if k in STOCK else p
        st = np.minimum(np.maximum(new, 0.0), CEIL)
        st[dead] = 0.0
    return st.reshape(N_AGES, n, N_OUT)


if RUN_ROLLOUT:
    Xr, Yr = sample_rows(TRAIN, ROLL_ROWS)
    Xv, Yv = sample_rows(HOLD, ROLL_VAL_ROWS)
    A_hold, M_hold = A_all[HOLD], M_all[HOLD]        # all the rest of the grid arrays is no longer needed: free RAM
    del A_all, M_all, y_all; gc.collect()
    stamp(f'rollout rows: train {Xr.shape} | holdout {Xv.shape}')
    # binned once (same params as training, so lgb.train does not need to rebuild it); only the label changes per target
    dtr = lgb.Dataset(Xr, Yr[:, 0], feature_name=RNAMES, params=PARAMS, free_raw_data=False).construct()
    dva = lgb.Dataset(Xv, Yv[:, 0], reference=dtr, params=PARAMS, free_raw_data=False).construct()
    rmodels = {}
    for k in range(N_OUT):
        dtr.set_label(Yr[:, k]); dva.set_label(Yv[:, k])
        rmodels[k] = lgb.train(PARAMS, dtr, ROLL_ROUNDS, valid_sets=[dva], valid_names=['hold'],
                               callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(500)])
        gi, top = importance(rmodels[k])
        SUMMARY.setdefault('rollout', {})[STATE_NAMES[k]] = dict(best_iter=rmodels[k].best_iteration, groups=gi, top=top)
        stamp(f'rollout {STATE_NAMES[k]}: best iter {rmodels[k].best_iteration} | gain by group {gi}')
    del dtr, dva, Xr, Yr, Xv, Yv; gc.collect()
    hp = rollout(rmodels, hold_y0, A_hold, M_hold)
    SUMMARY['rollout']['hold_metric'] = float(metric(hp, hold_y40))
    stamp(f'rollout holdout metric {metric(hp, hold_y40):.4f} (persistence {SUMMARY["hold_persistence"]:.4f})')
    for v in VARIANTS:
        key = f'rollout_{v}'
        PREDS[key] = {}
        for scn, T in TEST.items():
            PREDS[key][scn] = rollout(rmodels, T['y0'][v], T['A'][v], T['M'][v])
            stamp(f'rollout {key} {scn} done')
        write_submission(PREDS[key], f'{OUT}/submission_gbm_{key}{TAG}.csv')
        report(key)
    save_outputs()

# %% how different are the tree models from each other (the blend needs diversity, INSIGHTS 4.21)
keys = sorted(PREDS)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = PREDS[keys[i]], PREDS[keys[j]]
        stamp(f'msd {keys[i]} vs {keys[j]}: {np.mean([metric(a[s], b[s]) for s in SCENARIOS]):.4f}')
SUMMARY['total_min'] = (time.time() - T0) / 60
save_outputs()
