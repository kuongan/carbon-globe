# %% [markdown]
# CarbonGlobe: annual GRU emulator (Kaggle GPU, PyTorch, no repo clone). PLAN track C1.
#
# A different model family for the blend: it steps through the years like ED itself, state(t) + climate(t) ->
# state(t+1), 40 times, instead of mapping y0 + a 40-year climate summary to year 40 in one step. The 0.060 public
# team's version scored 0.144 alone but improved their blend (0.0645 -> 0.0622) because its errors differ.
#   per year : 12 months of f0-f6 + annual means of the 111 non-CO2 drivers -> climate embedding
#   context  : the site's 40-year mean climate + its chronosequence (y0 height / agb of all 15 seed ages)
#   state    : height, agb, soil, lai (standardized); GRU cell per (site, age); state(t+1) = state(t) + head(h, state)
#   loss     : every year of the free rollout (no teacher forcing), year 40 x5, height / agb in metric units
# Same 10-degree block holdout as the direct NN (HOLD_SEED 42). Test variants y0inv2 / y0inv / none, forcing as given.
# Inputs: competition data, zhihaow/carbonglobe, and a dataset holding test_maps.npz.
# Outputs (/kaggle/working): emu_preds.npz (PHYSICAL (15, N, 7) per <scn>_<variant>, seed mean and _s<k>),
# emu_summary.json. Locally EMU_SMOKE=1 runs 3000 sites.

# %%
import os, time, gc, json, glob, math
import numpy as np
import torch
import torch.nn as nn

T0 = time.time()
SMOKE = os.environ.get('EMU_SMOKE', '0') == '1'
if os.path.isdir('/kaggle/input'):
    COMP = '/kaggle/input/competitions/ieee-bigdata-cup-2026-ai-emulation-challenge/public'
    BASE = '/kaggle/input/datasets/zhihaow/carbonglobe'
    OUT = '/kaggle/working'
    MAPS = sorted(glob.glob('/kaggle/input/**/test_maps.npz', recursive=True))[0]
else:
    COMP, BASE, OUT = 'data/public', 'data/carbonglobe', 'result/emu_smoke'
    MAPS = 'kaggle_upload/noco2-ep4/test_maps.npz'
os.makedirs(OUT, exist_ok=True)

SEEDS = [int(s) for s in os.environ.get('EMU_SEEDS', '0,1,2').split(',')]
WEIGHTS = os.environ.get('EMU_WEIGHTS', '')            # dir with emu_seed<k>.pt: inference only, no training
EPOCHS = int(os.environ.get('EMU_EPOCHS', 25))
BS_SITES, LR, WD = 256, 2e-3, 1e-4
EMB, HID = 128, 256
W40 = 5.0
VARIANTS = ['y0inv2', 'y0inv', 'none']
HOLD_FRAC, BLOCK_DEG, HOLD_SEED = 0.08, 10, 42
SEED_AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
SCENARIOS = ['ssp126', 'ssp585']
NA, NY, NT, NC, NO = 15, 40, 12, 136, 7
ST = [0, 1, 2, 3]                                             # states rolled: height, agb, soil, lai
H_SCALE, A_SCALE = 10.90645, 3.39567
CO2 = list(range(8, 32))
KEEP = [j for j in range(NC) if j not in CO2 and j != 32]
MON = list(range(7))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stamp(m):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {m}', flush=True)


stats = np.load(f'{BASE}/data_stats/data_stats.npz')
X_MEAN, X_STD = stats['x_mean'].astype(np.float32), stats['x_std'].astype(np.float32)
Y_MEAN, Y_STD = stats['y_mean'].astype(np.float32), stats['y_std'].astype(np.float32)
_mz = np.load(MAPS)
MAP_XS, MAP_XB = _mz['x_s'].astype(np.float32), _mz['x_b'].astype(np.float32)
MAP_YS, MAP_YB = _mz['y_s'].astype(np.float32), _mz['y_b'].astype(np.float32)
stamp(f'torch {torch.__version__} | {DEV} | seeds {SEEDS} | epochs {EPOCHS} | smoke {SMOKE}')


def clim_year(x):
    """raw (n, 40, 12, 136) -> per-year climate (n, 40, 84 + 111), standardized and clipped"""
    mon = (x[..., MON] - X_MEAN[MON]) / (X_STD[MON] + 1e-10)
    ann = (x.mean(2)[..., KEEP] - X_MEAN[KEEP]) / (X_STD[KEEP] + 1e-10)
    return np.clip(np.concatenate([mon.reshape(len(x), NY, -1), ann], -1), -10, 10).astype(np.float32)


def norm_state(y):
    return ((y[..., ST] - Y_MEAN[ST]) / Y_STD[ST]).astype(np.float32)


def context(clim, y0):
    """site context: 40-year mean of the per-year climate + chronosequence (y0 height/agb of the 15 ages, standardized)"""
    prof = ((y0[..., :2] - Y_MEAN[:2]) / Y_STD[:2]).transpose(1, 0, 2).reshape(y0.shape[1], 30)
    return np.concatenate([clim.mean(1), prof], 1).astype(np.float32)


# %% grid
D = f'{BASE}/data_global'
gx = np.load(f'{D}/glob_X_fea.npy', mmap_mode='r')
SITES = np.arange(0, gx.shape[0], gx.shape[0] // 3000) if SMOKE else np.arange(gx.shape[0])
N = len(SITES)
CL = np.empty((N, NY, 84 + len(KEEP)), np.float32)
BAD = np.zeros(N, bool)
for s in range(0, N, 1024):
    c = np.array(gx[SITES[s:s + 1024]] if SMOKE else gx[s:s + 1024], np.float32)
    BAD[s:s + 1024] = ~np.isfinite(c).all(axis=(1, 2, 3))
    CL[s:s + 1024] = clim_year(np.nan_to_num(c))
    del c
del gx; gc.collect()
Y = np.empty((NA, N, NY + 1, NO), np.float32)
for ai, age in enumerate(SEED_AGES):
    g = np.load(f'{D}/glob_Y{age:03d}.npy', mmap_mode='r')
    for s in range(0, N, 8192):
        Y[ai, s:s + 8192] = (g[SITES[s:s + 8192]] if SMOKE else g[s:s + 8192])[:, :, 11, :]
    del g
OK = ~BAD & np.isfinite(Y).all(axis=(0, 2, 3))
Y = np.nan_to_num(Y)
CEIL = (2.0 * Y[:, OK].max(axis=(0, 1, 2))).astype(np.float32)
_r, _c = np.nonzero(np.load(f'{BASE}/data_stats/mask2d.npy') > 0)
_cells = int(round(BLOCK_DEG / 0.5))
BLOCK = ((_r // _cells) * 100000 + (_c // _cells))[SITES]
_ub = np.unique(BLOCK[OK])
_hb = np.random.default_rng(HOLD_SEED).choice(_ub, size=max(1, int(round(HOLD_FRAC * len(_ub)))), replace=False)
TRAIN, HOLD = np.flatnonzero(OK & ~np.isin(BLOCK, _hb)), np.flatnonzero(OK & np.isin(BLOCK, _hb))
CTX = context(CL, Y[:, :, 0])
SN = norm_state(Y)                                                    # (15, N, 41, 4)
stamp(f'grid: {N} sites, train {len(TRAIN)}, holdout {len(HOLD)}')


def metric(p, t):
    return 0.5 * (np.mean(((p[..., 0] - t[..., 0]) / H_SCALE) ** 2) + np.mean(((p[..., 1] - t[..., 1]) / A_SCALE) ** 2))


# %% test inputs (forcing as given; y0 variants as in the direct models)
TEST = {}
for scn in SCENARIOS:
    d = np.load(f'{COMP}/test_{scn}.npz')
    clt = clim_year(d['test_x'].astype(np.float32))
    y0t = np.transpose(d['test_y0'].astype(np.float32), (1, 0, 2))
    y0p = np.maximum((y0t - MAP_YB) / MAP_YS, 0.0).astype(np.float32)
    y0x = np.where(y0p[..., :1] > 0, np.maximum(2 * y0p - y0t, 0.0), 0.0).astype(np.float32)
    TEST[scn] = dict(cl=clt, y0={'y0inv': y0p, 'none': y0t, 'y0inv2': y0x})
    stamp(f'{scn}: test ready')


# %% model
class Emulator(nn.Module):
    def __init__(self):
        super().__init__()
        nf, nx = CL.shape[2], CTX.shape[1]
        self.clim = nn.Sequential(nn.Linear(nf, EMB), nn.SiLU(), nn.Linear(EMB, EMB), nn.SiLU())
        self.ctx = nn.Sequential(nn.Linear(nx, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        ns = len(ST)
        self.h0 = nn.Linear(64 + ns + 1, HID)
        self.cell = nn.GRUCell(EMB + 64 + ns + 1, HID)
        self.head = nn.Sequential(nn.Linear(HID + ns, 128), nn.SiLU(), nn.Linear(128, ns))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)      # starts as persistence

    def forward(self, cl, ctx, s0):
        """cl (B, 40, F), ctx (B, X), s0 (B, 15, 4) standardized -> trajectory (B, 15, 40, 4)"""
        B = cl.shape[0]
        e = self.clim(cl).unsqueeze(1).expand(B, NA, NY, EMB).reshape(B * NA, NY, EMB)
        c = self.ctx(ctx).unsqueeze(1).expand(B, NA, 64).reshape(B * NA, 64)
        la = torch.log(torch.tensor(SEED_AGES, device=cl.device, dtype=torch.float32)).repeat(B)[:, None] / 6.2
        s = s0.reshape(B * NA, -1)
        h = torch.tanh(self.h0(torch.cat([c, s, la], 1)))
        out = []
        for t in range(NY):
            at = torch.log(torch.exp(la * 6.2) + t) / 6.2
            h = self.cell(torch.cat([e[:, t], c, s, at], 1), h)
            s = s + self.head(torch.cat([h, s], 1))
            out.append(s)
        return torch.stack(out, 1).reshape(B, NA, NY, -1)


SNM = torch.tensor(Y_MEAN[ST], device=DEV); SNS = torch.tensor(Y_STD[ST], device=DEV)
SCL = torch.tensor([Y_STD[0] / H_SCALE, Y_STD[1] / A_SCALE], device=DEV)        # standardized -> metric units
WT = torch.ones(NY, device=DEV); WT[-1] = W40


def to_phys(sn):
    return sn * SNS + SNM


def rollout_phys(net, cl, ctx, y0, bs=512):
    """numpy in, physical year-40 state (15, n, 7) out; dead stands stay dead, clipped to [0, ceiling]"""
    n = cl.shape[0]
    out = y0.copy()
    s0n = norm_state(y0)
    net.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            tr = net(torch.from_numpy(cl[sl]).to(DEV), torch.from_numpy(ctx[sl]).to(DEV),
                     torch.from_numpy(np.ascontiguousarray(s0n[:, sl].transpose(1, 0, 2))).to(DEV))
            p = to_phys(tr[:, :, -1]).cpu().numpy().transpose(1, 0, 2)                    # (15, b, 4)
            for k, j in enumerate(ST):
                out[:, sl, j] = np.clip(p[..., k], 0, CEIL[j])
    dead = (y0[..., 0] == 0) & (y0[..., 1] == 0)
    out[dead] = 0.0
    return out


def train_one(seed, max_steps=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Emulator().to(DEV)
    CLT, CTXT = torch.from_numpy(CL).to(DEV), torch.from_numpy(CTX).to(DEV)
    SNT = torch.from_numpy(SN).to(DEV)                                                   # (15, N, 41, 4)
    tr = torch.from_numpy(TRAIN).to(DEV)
    steps_ep = math.ceil(len(TRAIN) / BS_SITES)
    total = max_steps or EPOCHS * steps_ep
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.1)
    best, best_state, hist, step = np.inf, None, [], 0
    for ep in range(EPOCHS):
        net.train()
        perm = tr[torch.randperm(len(tr), device=DEV)]
        tl = 0.0
        for i in range(0, len(perm), BS_SITES):
            s = perm[i:i + BS_SITES]
            traj = SNT[:, s].transpose(0, 1)                                              # (B, 15, 41, 4)
            pred = net(CLT[s], CTXT[s], traj[:, :, 0])
            err = (pred - traj[:, :, 1:])                                                 # (B, 15, 40, 4)
            e2 = (err[..., :2] * SCL) ** 2                                                # metric units
            loss = (e2.mean((0, 1, 3)) * WT).sum() / WT.sum() + 0.1 * (err[..., 2:] ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tl += float(loss.detach()); step += 1
            if step >= total:
                break
        hp = rollout_phys(net, CL[HOLD], CTX[HOLD], Y[:, HOLD, 0])
        hm = float(metric(hp, Y[:, HOLD, -1]))
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(1, steps_ep), hold_metric=hm))
        stamp(f'seed {seed} epoch {ep + 1}/{EPOCHS}: train {hist[-1]["train_loss"]:.5f} | holdout year-40 {hm:.5f}')
        if hm < best:
            best, best_state = hm, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if step >= total:
            break
    net.load_state_dict(best_state)
    return net, dict(seed=seed, best_hold=best, history=hist)


def test_predict(net, sites=None):
    res = {}
    for scn, T in TEST.items():
        cl = T['cl'] if sites is None else T['cl'][sites]
        for v in VARIANTS:
            y0 = T['y0'][v] if sites is None else T['y0'][v][:, sites]
            res[f'{scn}_{v}'] = rollout_phys(net, cl, context(cl, y0), y0)
    return res


# %% pre-flight, then seeds
SUMMARY = dict(config=dict(seeds=SEEDS, epochs=EPOCHS, emb=EMB, hid=HID, w40=W40, variants=VARIANTS,
                           n_train=int(len(TRAIN)), n_hold=int(len(HOLD))),
               hold_persistence=float(metric(Y[:, HOLD, 0], Y[:, HOLD, -1])), seeds=[])
if not SMOKE and not WEIGHTS:
    _e = EPOCHS; EPOCHS = 1
    net, _ = train_one(123, max_steps=20)
    EPOCHS = _e
    pf = test_predict(net, sites=np.arange(0, 6171, 500))
    assert all(np.isfinite(v).all() for v in pf.values()), 'non-finite test predictions'
    stamp('pre-flight ok'); del net
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()

PREDS = {}
for seed in SEEDS:
    if WEIGHTS:
        net = Emulator().to(DEV)
        net.load_state_dict(torch.load(f'{WEIGHTS}/emu_seed{seed}.pt', map_location=DEV))
        info = dict(seed=seed, best_hold=float('nan'), history=[], loaded=True)
    else:
        net, info = train_one(seed)
        torch.save(net.state_dict(), f'{OUT}/emu_seed{seed}.pt')
    SUMMARY['seeds'].append(info)
    for k, v in test_predict(net).items():
        PREDS[f'{k}_s{seed}'] = v
    np.savez_compressed(f'{OUT}/emu_preds.npz', **PREDS)
    stamp(f'seed {seed}: best holdout {info["best_hold"]:.5f} (persistence {SUMMARY["hold_persistence"]:.5f}) | saved')
    del net
    if DEV.type == 'cuda':
        torch.cuda.empty_cache()
done = [i['seed'] for i in SUMMARY['seeds']]
for scn in SCENARIOS:
    for v in VARIANTS:
        PREDS[f'{scn}_{v}'] = np.mean([PREDS[f'{scn}_{v}_s{s}'] for s in done], axis=0)
np.savez_compressed(f'{OUT}/emu_preds.npz', **PREDS)
with open(f'{OUT}/emu_summary.json', 'w') as f:
    json.dump(SUMMARY, f, indent=1)
stamp(f'done: emu_preds.npz ({len(PREDS)} arrays), emu_summary.json')
