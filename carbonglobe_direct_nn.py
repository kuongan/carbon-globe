# %% [markdown]
# CarbonGlobe: neural DIRECT model on the global grid (Kaggle GPU, PyTorch, no repo clone). INSIGHTS 4.36.
#
# A new member family for the blend. Like the LightGBM direct model (y0 + 40-year climate -> year-40 state in one
# step, no rollout, so no compounding error), but a neural net: smooth in its inputs, so it extrapolates on the
# shifted test climate instead of going flat like trees, and its errors differ from both the LSTMs and the trees.
#   site encoder : 2 x Conv1d + bidirectional GRU over the 40 annual means of the 111 non-CO2 features
#   + monthly climatology of f0-f6, y0 (7 variables, standardized), seed age (one-hot + age/500)
#   -> change of height & agb at years 10/20/30/40 (years 10-30 are auxiliary targets, year 40 is scored)
# Trained on physical grid data (CO2 block and the constant feature dropped, like the no-CO2 models). Several seeds,
# averaged. Test variants (as in carbonglobe_gbm.py; each model gives y0 pairs like the ones behind x1 / x2):
#   y0inv  y0 physical, forcing as given (the GBM's best)    none   y0 left in the test space, forcing as given
#   y0inv2 y0 pushed past physical (2 y0_p - y0_t)           inv    y0 and forcing physical      xinv  y0 test, forcing physical
#
# Inputs: competition data, zhihaow/carbonglobe, and a dataset holding test_maps.npz (noco2-ep4 or lstm-sweep).
# Outputs (/kaggle/working): nn_preds.npz (PHYSICAL (15, N, 7); keys <scn>_<variant> = seed mean, <scn>_<variant>_s<k>),
# nn_summary.json, submission_nn_<variant>.csv (mapped with the test_maps slopes, not mean-matched: done locally).
# Locally without PyTorch, NN_SMOKE=1 runs the whole data pipeline with a dummy model (outputs in result/nn_smoke/).

# %%
import os, time, gc, json, glob, math
import numpy as np
import pandas as pd

T0 = time.time()
SMOKE = os.environ.get('NN_SMOKE', '0') == '1'
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:                                  # local smoke test of the data pipeline only
    HAVE_TORCH = False
    assert SMOKE, 'PyTorch is needed outside NN_SMOKE=1'

if os.path.isdir('/kaggle/input'):
    COMP = '/kaggle/input/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public'
    BASE = '/kaggle/input/datasets/zhihaow/carbonglobe'
    OUT = '/kaggle/working'
    MAPS = sorted(glob.glob('/kaggle/input/**/test_maps.npz', recursive=True))[0]
else:
    COMP, BASE, OUT = 'data/public', 'data/carbonglobe', 'result/nn_smoke'
    MAPS = 'kaggle_upload/noco2-ep4/test_maps.npz'
os.makedirs(OUT, exist_ok=True)

# ---- config ----
SEEDS = [int(s) for s in os.environ.get('NN_SEEDS', '0,1,2,3').split(',')]
EPOCHS = int(os.environ.get('NN_EPOCHS', 20))
BS, LR, WD, DROPOUT = 4096, 2e-3, 1e-4, 0.1
D_MODEL, HIDDEN = 128, 512
HORIZONS = [10, 20, 30, 40]                          # predicted change at these years; year 40 is scored
H_WEIGHT = [0.25, 0.25, 0.25, 1.0]
VARIANTS = ['y0inv', 'none', 'y0inv2', 'inv', 'xinv']
HOLD_FRAC, BLOCK_DEG, HOLD_SEED = 0.08, 10, 42       # same 10-degree block holdout for every seed
STATE_CEIL = 2.0                                     # predictions capped at 2x the grid maximum
PREFLIGHT = not SMOKE                                # a few steps + every test variant on a few sites, before the long run
SMOKE_SITES = 3000

SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCENARIOS = ['ssp126', 'ssp585']
N_AGES, N_YEAR, N_T, N_CONS, N_OUT = len(SEED_AGES), 40, 12, 136, 7
H_SCALE, A_SCALE = 10.90645, 3.39567
CO2 = list(range(8, 32))
KEEP = [j for j in range(N_CONS) if j not in CO2 and j != 32]   # 111 features
MON = list(range(7))                                             # f0-f6 monthly climatology (f7: annual mean only)
N_SEQ, N_CLIM, N_AGEF = len(KEEP), N_T * len(MON), N_AGES + 1
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAVE_TORCH else None


def stamp(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


stamp(f'torch {torch.__version__ if HAVE_TORCH else "-"} | device {DEV} | seeds {SEEDS} | epochs {EPOCHS} | smoke {SMOKE}')
stats = np.load(f'{BASE}/data_stats/data_stats.npz')
X_MEAN, X_STD = stats['x_mean'].astype(np.float32), stats['x_std'].astype(np.float32)
Y_MEAN, Y_STD = stats['y_mean'].astype(np.float32), stats['y_std'].astype(np.float32)
_mz = np.load(MAPS)
MAP_XS, MAP_XB = _mz['x_s'].astype(np.float32), _mz['x_b'].astype(np.float32)
MAP_YS, MAP_YB = _mz['y_s'].astype(np.float32), _mz['y_b'].astype(np.float32)


def seq_feats(A):
    """annual means (n, 40, 136) raw -> standardized (n, 40, 111), clipped like the LSTM inputs"""
    return np.clip((A[:, :, KEEP] - X_MEAN[KEEP]) / (X_STD[KEEP] + 1e-10), -10, 10).astype(np.float32)


def clim_feats(M):
    """monthly f0-f6 (n, 40, 12, 7) raw -> 40-year monthly climatology, standardized, (n, 84)"""
    c = (M.mean(axis=1) - X_MEAN[MON]) / (X_STD[MON] + 1e-10)
    return np.clip(c, -10, 10).reshape(len(M), -1).astype(np.float32)


AGE_F = np.concatenate([np.eye(N_AGES), np.array(SEED_AGES)[:, None] / 500.0], 1).astype(np.float32)   # (15, 16)


def y0_norm(y0):
    return ((y0 - Y_MEAN) / Y_STD).astype(np.float32)


# %% Load the grid (as carbonglobe_gbm.py): annual means, monthly f0-f6, annual states (month 11) of the 15 ages
D = f'{BASE}/data_global'
gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
SITES = np.arange(0, gx.shape[0], max(1, gx.shape[0] // SMOKE_SITES)) if SMOKE else np.arange(gx.shape[0])
N_G = len(SITES)
SEQ = np.empty((N_G, N_YEAR, N_SEQ), np.float32)
CLIM = np.empty((N_G, N_CLIM), np.float32)
X_BAD = np.zeros(N_G, bool)
FULL = not SMOKE                                     # all sites: contiguous slices read faster than index arrays
for s in range(0, N_G, 1024):
    c = np.array(gx[s:s + 1024] if FULL else gx[SITES[s:s + 1024]], dtype=np.float32)
    X_BAD[s:s + 1024] = ~np.isfinite(c).all(axis=(1, 2, 3))
    c = np.nan_to_num(c)
    SEQ[s:s + 1024] = seq_feats(c.mean(axis=2))
    CLIM[s:s + 1024] = clim_feats(c[..., MON])
    del c
    if (s // 1024) % 10 == 0:
        stamp(f'glob_X {s + 1024}/{N_G}')
del gx; gc.collect()
Y = np.empty((N_AGES, N_G, len(HORIZONS) + 1, N_OUT), np.float32)          # years 0, 10, 20, 30, 40
for ai, age in enumerate(SEED_AGES):
    g = np.load(f'{D}/glob_Y{age:03d}.npy', mmap_mode='r')
    for s in range(0, N_G, 8192):
        blk = g[s:s + 8192] if FULL else g[SITES[s:s + 8192]]
        Y[ai, s:s + 8192] = blk[:, [0] + HORIZONS, 11, :]
    del g
SITE_OK = ~X_BAD & np.isfinite(Y).all(axis=(0, 2, 3))
CEIL = (STATE_CEIL * np.nanmax(Y[:, SITE_OK], axis=(0, 1, 2))).astype(np.float32)
Y0N = y0_norm(Y[:, :, 0])                                                   # (15, N, 7)
SC = np.array([H_SCALE, A_SCALE], np.float32)
TGT = ((Y[:, :, 1:, :2] - Y[:, :, :1, :2]) / SC).reshape(N_AGES, N_G, -1)   # (15, N, 8): change in metric units
Y40, Y0 = Y[:, :, -1], Y[:, :, 0]
del Y; gc.collect()

_r, _c = np.nonzero(np.load(f'{BASE}/data_stats/mask2d.npy') > 0)
_cells = int(round(BLOCK_DEG / 0.5))
BLOCK = ((_r // _cells) * 100000 + (_c // _cells))[SITES]
_ub = np.unique(BLOCK[SITE_OK])
_hb = np.random.default_rng(HOLD_SEED).choice(_ub, size=max(1, int(round(HOLD_FRAC * len(_ub)))), replace=False)
IS_HOLD = np.isin(BLOCK, _hb)
TRAIN, HOLD = np.flatnonzero(SITE_OK & ~IS_HOLD), np.flatnonzero(SITE_OK & IS_HOLD)
stamp(f'grid: sites {N_G} (usable {SITE_OK.sum()}) | train {len(TRAIN)} | holdout {len(HOLD)} | ceiling {np.round(CEIL[:2], 2)}')


def metric(p, t):
    return 0.5 * (np.mean(((p[..., 0] - t[..., 0]) / H_SCALE) ** 2) + np.mean(((p[..., 1] - t[..., 1]) / A_SCALE) ** 2))


# %% Test inputs per variant (test = s * physical + b; maps from test_maps.npz)
X_KEEP = np.ones(N_CONS, bool); X_KEEP[CO2] = False; X_KEEP[32] = False
TEST = {}
for scn in SCENARIOS:
    d = np.load(f'{COMP}/test_{scn}.npz')
    tx = d['test_x'].astype(np.float32)
    y0_t = np.transpose(d['test_y0'].astype(np.float32), (1, 0, 2))                # (15, N, 7) test space
    y0_p = np.maximum((y0_t - MAP_YB) / MAP_YS, 0.0).astype(np.float32)          # physical; dead -> exactly 0
    y0_x = np.where(y0_p[..., :1] > 0, np.maximum(2 * y0_p - y0_t, 0.0), 0.0).astype(np.float32)
    F = {'T': (seq_feats(tx.mean(axis=2)), clim_feats(tx[..., MON]))}
    for j in np.flatnonzero(X_KEEP):                                              # forcing mapped back, in place
        tx[..., j] -= MAP_XB[j]; tx[..., j] /= MAP_XS[j]
    F['P'] = (seq_feats(tx.mean(axis=2)), clim_feats(tx[..., MON]))
    Y0V = {'y0inv': y0_p, 'none': y0_t, 'y0inv2': y0_x, 'inv': y0_p, 'xinv': y0_t}
    FV = {'y0inv': 'T', 'none': 'T', 'y0inv2': 'T', 'inv': 'P', 'xinv': 'P'}
    TEST[scn] = dict(F=F, Y0=Y0V, FV=FV)
    del tx, d; gc.collect()
    stamp(f'{scn}: test inputs ready | dead share {np.mean(y0_p[..., 0] == 0):.4f}')


# %% Model
if HAVE_TORCH:
    class DirectNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = D_MODEL
            self.conv = nn.Sequential(nn.Conv1d(N_SEQ, d, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(d, d, 3, padding=1), nn.GELU())
            self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
            self.site = nn.Sequential(nn.Linear(4 * d, 256), nn.GELU(), nn.LayerNorm(256))
            self.clim = nn.Sequential(nn.Linear(N_CLIM, 64), nn.GELU())
            self.head = nn.Sequential(nn.Linear(256 + 64 + N_OUT + N_AGEF, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                                      nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                                      nn.Linear(HIDDEN, 2 * len(HORIZONS)))

        def forward(self, seq, clim, y0, age):                      # seq (B, 40, N_SEQ)
            h = self.conv(seq.transpose(1, 2)).transpose(1, 2)
            g, _ = self.gru(h)
            s = self.site(torch.cat([g.mean(1), g[:, -1]], -1))
            return self.head(torch.cat([s, self.clim(clim), y0, age], -1))


def batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


def predict_delta(net, seq, clim, y0n):
    """seq (n, 40, F), clim (n, 84) per site; y0n (15, n, 7) -> change at year 40, metric units (15, n, 2)"""
    n = seq.shape[0]
    out = np.zeros((N_AGES, n, 2), np.float32)
    if net is None:                                                 # smoke test without PyTorch: dummy model
        return out
    net.eval()
    S, Cl = torch.from_numpy(seq).to(DEV), torch.from_numpy(clim).to(DEV)
    with torch.no_grad(), torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == 'cuda'):
        for a in range(N_AGES):
            Y0a = torch.from_numpy(y0n[a]).to(DEV)
            Aa = torch.from_numpy(AGE_F[a]).to(DEV)
            for b in batches(n, 8192):
                o = net(S[b], Cl[b], Y0a[b], Aa.expand(b.stop - b.start, -1)).float()
                out[a, b] = o[:, -2:].cpu().numpy()
    return out


def to_state(y0, delta):
    """y0 (15, n, 7) as fed, change (15, n, 2) -> predicted year-40 state (15, n, 7); dead stays dead"""
    p = y0.copy()
    dead = (y0[..., 0] == 0) & (y0[..., 1] == 0)
    for k in (0, 1):
        p[..., k] = np.where(dead, 0.0, np.clip(y0[..., k] + delta[..., k] * SC[k], 0.0, CEIL[k]))
    return p


def test_predict(net, sites=None):
    res = {}
    for scn, T in TEST.items():
        for v in VARIANTS:
            seq, clim = T['F'][T['FV'][v]]
            y0 = T['Y0'][v]
            if sites is not None:
                seq, clim, y0 = seq[sites], clim[sites], y0[:, sites]
            res[f'{scn}_{v}'] = to_state(y0, predict_delta(net, seq, clim, y0_norm(y0)))
    return res


def train_one(seed, max_steps=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = DirectNet().to(DEV)
    S, Cl = torch.from_numpy(SEQ).to(DEV), torch.from_numpy(CLIM).to(DEV)
    Y0T, TG = torch.from_numpy(Y0N).to(DEV), torch.from_numpy(TGT).to(DEV)
    AG = torch.from_numpy(AGE_F).to(DEV)
    W = torch.tensor([w for w in H_WEIGHT for _ in (0, 1)], device=DEV)
    tr = torch.from_numpy(np.stack(np.meshgrid(np.arange(N_AGES), TRAIN, indexing='ij'), -1).reshape(-1, 2)).to(DEV)
    n_tr = tr.shape[0]
    steps_ep = math.ceil(n_tr / BS)
    total = max_steps or EPOCHS * steps_ep
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=DEV.type == 'cuda')
    best, best_state, hist, step = np.inf, None, [], 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n_tr, device=DEV)
        tl = 0.0
        for b in batches(n_tr, BS):
            idx = tr[perm[b]]
            a, s = idx[:, 0], idx[:, 1]
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == 'cuda'):
                out = net(S[s], Cl[s], Y0T[a, s], AG[a])
            loss = (W * (out.float() - TG[a, s]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss); step += 1
            if step >= total:
                break
        hp = to_state(Y0[:, HOLD], predict_delta(net, SEQ[HOLD], CLIM[HOLD], Y0N[:, HOLD]))
        hm = float(metric(hp, Y40[:, HOLD]))
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(1, step - ep * steps_ep), hold_metric=hm))
        stamp(f'seed {seed} epoch {ep + 1}/{EPOCHS}: train {hist[-1]["train_loss"]:.5f} | holdout year-40 metric {hm:.5f}')
        if hm < best:
            best, best_state = hm, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if step >= total:
            break
    net.load_state_dict(best_state)
    return net, dict(seed=seed, best_hold=best, history=hist)


# %% Pre-flight (a few steps, every variant on a few test sites), then the seeds
SUMMARY = dict(config=dict(seeds=SEEDS, epochs=EPOCHS, bs=BS, lr=LR, d_model=D_MODEL, hidden=HIDDEN,
                           horizons=HORIZONS, variants=VARIANTS, n_train=int(len(TRAIN)), n_hold=int(len(HOLD))),
               hold_persistence=float(metric(Y0[:, HOLD], Y40[:, HOLD])), seeds=[])
if HAVE_TORCH and PREFLIGHT:
    _ep = EPOCHS; EPOCHS = 1
    net, _ = train_one(123, max_steps=20)
    EPOCHS = _ep
    pf = test_predict(net, sites=np.arange(0, 6171, 500))
    assert all(np.isfinite(v).all() for v in pf.values()), 'non-finite test predictions'
    stamp(f'pre-flight ok: {len(pf)} test arrays, e.g. {next(iter(pf))} {next(iter(pf.values())).shape}')
    del net; torch.cuda.empty_cache()

PREDS = {}
for seed in (SEEDS if HAVE_TORCH else [0]):
    if HAVE_TORCH:
        net, info = train_one(seed)
    else:
        net, info = None, dict(seed=seed, best_hold=float('nan'), history=[])
    SUMMARY['seeds'].append(info)
    res = test_predict(net)
    for k, v in res.items():
        PREDS[f'{k}_s{seed}'] = v
    np.savez_compressed(f'{OUT}/nn_preds.npz', **PREDS)                       # saved after every seed
    stamp(f'seed {seed}: best holdout {info["best_hold"]:.5f} (persistence {SUMMARY["hold_persistence"]:.5f}) | saved')
    if HAVE_TORCH:
        del net; torch.cuda.empty_cache()

done = [i['seed'] for i in SUMMARY['seeds']]
for scn in SCENARIOS:
    for v in VARIANTS:
        PREDS[f'{scn}_{v}'] = np.mean([PREDS[f'{scn}_{v}_s{s}'] for s in done], axis=0)
np.savez_compressed(f'{OUT}/nn_preds.npz', **PREDS)
for v in VARIANTS:
    parts = []
    for scn in SCENARIOS:
        p = np.maximum(PREDS[f'{scn}_{v}'], 0.0) * MAP_YS + MAP_YB
        parts.append(pd.DataFrame({'id': [f'{scn}_{s}_{a}' for a in SEED_AGES for s in range(p.shape[1])],
                                   'height': (p[:, :, 0] / H_SCALE).reshape(-1), 'agb': (p[:, :, 1] / A_SCALE).reshape(-1)}))
    sub = pd.concat(parts, ignore_index=True)
    assert np.isfinite(sub[['height', 'agb']].to_numpy()).all()
    sub.to_csv(f'{OUT}/submission_nn_{v}.csv', index=False)
    m = {scn: tuple(np.round([PREDS[f'{scn}_{v}'][..., k].mean() / sc for k, sc in ((0, H_SCALE), (1, A_SCALE))], 3))
         for scn in SCENARIOS}
    stamp(f'{v}: physical means /scale {m}')
if len(done) > 1:
    SUMMARY['seed_msd_y0inv'] = float(np.mean([metric(PREDS[f'ssp126_y0inv_s{a}'], PREDS[f'ssp126_y0inv_s{b}'])
                                               for i, a in enumerate(done) for b in done[i + 1:]]))
    stamp(f'mean msd between seeds (ssp126 y0inv): {SUMMARY["seed_msd_y0inv"]:.5f}')
with open(f'{OUT}/nn_summary.json', 'w') as f:
    json.dump(SUMMARY, f, indent=1)
stamp(f'done: nn_preds.npz ({len(PREDS)} arrays), nn_summary.json, submission_nn_<variant>.csv')
