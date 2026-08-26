"""
Matched random controls: same hour, same weekday, same height in ATR,
same distance from price in ATR, same direction. Only difference: no
three-bar structure at that spot.
"""
import numpy as np, pandas as pd

RNG = np.random.RandomState(7)
K = 10   # controls per real zone

df = pd.read_pickle("/home/claude/es_5min.pkl").between_time("09:30", "16:00")
h = df["high"].values.astype("float64")
l = df["low"].values.astype("float64")
c = df["close"].values.astype("float64")
idx = df.index
n = len(df)
session = idx.normalize()

prev_c = np.concatenate([[c[0]], c[:-1]])
tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
atr = pd.Series(tr).rolling(14).mean().values

sess_end = pd.Series(np.arange(n)).groupby(session.values).transform("max").values

real = pd.read_pickle("/home/claude/fvg_real.pkl")
fvg = pd.read_pickle("/home/claude/fvg_zones.pkl")

HORIZONS = [1, 5, 20, 60, 240]
MAXH = max(HORIZONS)

hours = idx.hour.values
dows = idx.dayofweek.values
valid = np.isfinite(atr) & (atr > 0)

# pool of candidate bars per (hour, dow)
pools = {}
for hh in np.unique(hours):
    for dd in np.unique(dows):
        m = (hours == hh) & (dows == dd) & valid
        # leave room for the measurement horizon
        m &= (np.arange(n) < n - MAXH - 2)
        pools[(hh, dd)] = np.where(m)[0]

def measure(bar, direction, lo, hi, session_end_bar):
    if direction == 1:
        near, far = hi, lo
    else:
        near, far = lo, hi
    half = (hi + lo) / 2
    start = bar + 1
    end_all = min(n, start + MAXH)
    seg_l = l[start:end_all]; seg_h = h[start:end_all]
    if direction == 1:
        t_near = np.where(seg_l <= near)[0]; t_half = np.where(seg_l <= half)[0]; t_full = np.where(seg_l <= far)[0]
    else:
        t_near = np.where(seg_h >= near)[0]; t_half = np.where(seg_h >= half)[0]; t_full = np.where(seg_h >= far)[0]
    f_near = t_near[0] if len(t_near) else 10**9
    f_half = t_half[0] if len(t_half) else 10**9
    f_full = t_full[0] if len(t_full) else 10**9
    out = {}
    for H in HORIZONS:
        out[H] = (f_near < H, f_half < H, f_full < H)
    se = session_end_bar - bar
    out["session"] = (f_near < se, f_half < se, f_full < se)
    return out

rows = []
for r in fvg.itertuples(index=False):
    key = (idx[r.bar].hour, idx[r.bar].dayofweek)
    pool = pools.get(key)
    if pool is None or len(pool) < 20:
        continue
    picks = RNG.choice(pool, size=K, replace=False)
    for j in picks:
        if j == r.bar:
            continue
        aj = atr[j]
        height = r.height_atr * aj
        dist = r.dist_atr * aj
        ref = c[j]
        if r.dir == 1:          # zone below price
            near = ref - dist
            lo_, hi_ = near - height, near
        else:                   # zone above price
            near = ref + dist
            lo_, hi_ = near, near + height
        m = measure(j, r.dir, lo_, hi_, sess_end[j])
        row = {"src_bar": r.bar, "bar": j, "dir": r.dir,
               "height_atr": r.height_atr, "dist_atr": r.dist_atr,
               "hour": idx[j].hour, "session": str(session[j].date())}
        for H in HORIZONS + ["session"]:
            row[f"near_{H}"], row[f"half_{H}"], row[f"full_{H}"] = m[H]
        rows.append(row)

ctrl = pd.DataFrame(rows)
ctrl.to_pickle("/home/claude/fvg_ctrl.pkl")
print("control zones:", len(ctrl))

# ---------- headline comparison ----------
print("\n" + "="*66)
print("FILL RATES: real FVG vs matched random zones (RTH, ES, 2020-2025)")
print("="*66)
print(f"{'horizon':>10} | {'metric':>6} | {'FVG':>7} | {'random':>7} | {'diff':>7}")
print("-"*66)
for H in HORIZONS + ["session"]:
    for metric in ["near", "half", "full"]:
        a = real[f"{metric}_{H}"].mean() * 100
        b = ctrl[f"{metric}_{H}"].mean() * 100
        print(f"{str(H):>10} | {metric:>6} | {a:6.1f}% | {b:6.1f}% | {a-b:+6.1f}")
    print("-"*66)
