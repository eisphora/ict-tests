"""
FVG vs matched random zones. Methodology fixed in FVG_pre_registration.md.
"""
import numpy as np, pandas as pd

RNG = np.random.RandomState(20260821)

df = pd.read_pickle("/home/claude/es_5min.pkl")
# regular session only
df = df.between_time("09:30", "16:00")
print("bars in RTH:", len(df), df.index.min(), "->", df.index.max())

o = df["open"].values.astype("float64")
h = df["high"].values.astype("float64")
l = df["low"].values.astype("float64")
c = df["close"].values.astype("float64")
idx = df.index
n = len(df)
session = idx.normalize()

# ---- ATR(14) ----
prev_c = np.concatenate([[c[0]], c[:-1]])
tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
atr = pd.Series(tr).rolling(14).mean().values

# ---- detect FVG on bars (i-1, i, i+1); zone known at close of i+1 ----
recs = []
for i in range(1, n - 1):
    if session[i-1] != session[i+1]:
        continue                      # do not span the session break
    a = atr[i+1]
    if not np.isfinite(a) or a <= 0:
        continue
    if l[i+1] > h[i-1]:               # bullish
        lo, hi, direction = h[i-1], l[i+1], 1
    elif h[i+1] < l[i-1]:             # bearish
        lo, hi, direction = h[i+1], l[i-1], -1
    else:
        continue
    height = hi - lo
    ref = c[i+1]
    near = lo if direction == 1 else hi     # edge closest to price
    far = hi if direction == -1 else lo     # wait: define explicitly below
    # bullish gap sits BELOW price: near edge = hi (top of zone), far = lo
    if direction == 1:
        near, far = hi, lo
    else:
        near, far = lo, hi
    recs.append((i + 1, direction, lo, hi, height, height / a,
                 abs(ref - near) / a, ref, a))

fvg = pd.DataFrame(recs, columns=["bar", "dir", "lo", "hi", "height",
                                  "height_atr", "dist_atr", "ref", "atr"])
print("FVG found:", len(fvg))
print(fvg["height_atr"].describe())
fvg.to_pickle("/home/claude/fvg_zones.pkl")

# ---- fill measurement ----
HORIZONS = [1, 5, 20, 60, 240]
MAXH = max(HORIZONS)

def measure(bar, direction, lo, hi, session_end_bar):
    """returns dict horizon -> (touch_near, half, full) booleans"""
    if direction == 1:
        near, far = hi, lo
        half = (hi + lo) / 2
    else:
        near, far = lo, hi
        half = (hi + lo) / 2
    out = {}
    start = bar + 1
    end_all = min(n, start + MAXH)
    seg_l = l[start:end_all]
    seg_h = h[start:end_all]
    if direction == 1:
        t_near = np.where(seg_l <= near)[0]
        t_half = np.where(seg_l <= half)[0]
        t_full = np.where(seg_l <= far)[0]
    else:
        t_near = np.where(seg_h >= near)[0]
        t_half = np.where(seg_h >= half)[0]
        t_full = np.where(seg_h >= far)[0]
    f_near = t_near[0] if len(t_near) else 10**9
    f_half = t_half[0] if len(t_half) else 10**9
    f_full = t_full[0] if len(t_full) else 10**9
    for H in HORIZONS:
        out[H] = (f_near < H, f_half < H, f_full < H)
    # to session end
    se = session_end_bar - bar
    out["session"] = (f_near < se, f_half < se, f_full < se)
    return out

# session end bar index for each bar
sess_end = pd.Series(np.arange(n)).groupby(session.values).transform("max").values

rows = []
for r in fvg.itertuples(index=False):
    m = measure(r.bar, r.dir, r.lo, r.hi, sess_end[r.bar])
    row = {"bar": r.bar, "dir": r.dir, "height_atr": r.height_atr,
           "dist_atr": r.dist_atr, "hour": idx[r.bar].hour,
           "dow": idx[r.bar].dayofweek, "session": str(session[r.bar].date())}
    for H in HORIZONS + ["session"]:
        row[f"near_{H}"], row[f"half_{H}"], row[f"full_{H}"] = m[H]
    rows.append(row)

real = pd.DataFrame(rows)
real.to_pickle("/home/claude/fvg_real.pkl")
print("\nmeasured real zones:", len(real))
print("saved.")
