"""
Killzones vs all windows of equal length. Methodology fixed beforehand.
"""
import numpy as np, pandas as pd
from datetime import time

df = pd.read_pickle("/home/claude/es30s.pkl")
agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
m1 = df.resample("1min").agg(agg).dropna()
print("1-min bars:", len(m1), m1.index.min(), "->", m1.index.max())

m1["day"] = m1.index.normalize()
days = m1["day"].unique()
print("days:", len(days))

# ---- daily ATR on RTH, for normalisation ----
rth = m1.between_time("09:30", "16:00")
daily = rth.groupby("day").agg(hi=("high","max"), lo=("low","min"), close=("close","last"))
daily["range"] = daily["hi"] - daily["lo"]
daily["atr"] = daily["range"].rolling(14).mean()
print("median daily RTH range:", round(daily["range"].median(),1))

# minute-of-day index
m1["mod"] = m1.index.hour * 60 + m1.index.minute

ICT_WINDOWS = {
    "London KZ 02:00-05:00":      (120, 300),
    "NY AM KZ 07:00-10:00":       (420, 600),
    "NY PM KZ 13:30-16:00":       (810, 960),
    "Silver Bullet AM 10:00-11:00": (600, 660),
    "Silver Bullet PM 14:00-15:00": (840, 900),
    "Macro 09:50-10:10":          (590, 610),
    "Macro 08:50-09:10":          (530, 550),
}

def window_stats(start, end):
    """
    For each day compute metrics for window [start, end) in minutes-of-day.
    Returns dict of metric -> value aggregated across days.
    """
    seg = m1[(m1["mod"] >= start) & (m1["mod"] < end)]
    if len(seg) == 0:
        return None
    g = seg.groupby("day").agg(hi=("high","max"), lo=("low","min"),
                               first=("open","first"), last=("close","last"),
                               n=("close","size"))
    expected = end - start
    g = g[g["n"] >= expected * 0.6]          # день должен покрывать окно
    if len(g) < 200:
        return None
    g = g.join(daily[["hi","lo","atr","range"]], rsuffix="_day")
    g = g.dropna()
    if len(g) < 200:
        return None

    # 1. доля дневных экстремумов, попавших в окно
    tol = 1e-9
    hit_high = (g["hi"] >= g["hi_day"] - tol)
    hit_low  = (g["lo"] <= g["lo_day"] + tol)
    extreme_share = float((hit_high | hit_low).mean())

    # 2. средний ход, нормированный на ATR дня
    move = ((g["hi"] - g["lo"]) / g["atr"]).replace([np.inf,-np.inf], np.nan).dropna()
    avg_move = float(move.mean())

    # 3. направленность: |close-open| / range окна
    rng = (g["hi"] - g["lo"]).replace(0, np.nan)
    direct = (np.abs(g["last"] - g["first"]) / rng).replace([np.inf,-np.inf], np.nan).dropna()
    directionality = float(direct.mean())

    return {"extreme_share": extreme_share,
            "avg_move_atr": avg_move,
            "directionality": directionality,
            "days": int(len(g))}

# ---- continuation metric needs the hour after the window ----
def continuation(start, end):
    seg = m1[(m1["mod"] >= start) & (m1["mod"] < end)]
    nxt = m1[(m1["mod"] >= end) & (m1["mod"] < min(end + 60, 1440))]
    if len(seg) == 0 or len(nxt) == 0:
        return None
    a = seg.groupby("day").agg(first=("open","first"), last=("close","last"), n=("close","size"))
    b = nxt.groupby("day").agg(last2=("close","last"), n2=("close","size"))
    j = a.join(b, how="inner").dropna()
    j = j[(j["n"] >= (end-start)*0.6) & (j["n2"] >= 30)]
    if len(j) < 200:
        return None
    dir1 = np.sign(j["last"] - j["first"])
    dir2 = np.sign(j["last2"] - j["last"])
    mask = dir1 != 0
    if mask.sum() < 200:
        return None
    return float((dir1[mask] == dir2[mask]).mean())

# ---- build control distribution: all windows of same length, step 10 min ----
def control_distribution(length, step=10, lo=None, hi=None):
    rows = []
    lo = 0 if lo is None else lo
    hi = 1440 if hi is None else hi
    for s in range(lo, hi - length + 1, step):
        st = window_stats(s, s + length)
        if st is None:
            continue
        st["cont"] = continuation(s, s + length)
        st["start"] = s
        rows.append(st)
    return pd.DataFrame(rows)

print("\n" + "="*96)
print("KILLZONES vs ALL WINDOWS OF THE SAME LENGTH (ES, 2020-2025)")
print("="*96)

results = []
cache = {}
for name, (s, e) in ICT_WINDOWS.items():
    length = e - s
    if length not in cache:
        cache[length] = (control_distribution(length),
                         control_distribution(length, lo=480, hi=960))
        print(f"  [control {length} min: all-day {len(cache[length][0])}, session-only {len(cache[length][1])}]")
    pool, pool_sess = cache[length]
    st = window_stats(s, e)
    if st is None:
        print(f"{name}: not enough data")
        continue
    st["cont"] = continuation(s, e)

    row = {"window": name, "length_min": length, "days": st["days"]}
    for metric, key in [("доля дн. экстремумов","extreme_share"),
                        ("ход в ATR","avg_move_atr"),
                        ("направленность","directionality"),
                        ("продолжение","cont")]:
        val = st[key]
        col = pool[key].dropna()
        if val is None or len(col) == 0:
            row[key] = None; row[key+"_pct"] = None
            continue
        pct = float((col < val).mean() * 100)
        row[key] = val
        row[key+"_pct"] = pct
        cols = pool_sess[key].dropna() if key in pool_sess.columns else []
        row[key+"_pct_sess"] = float((cols < val).mean()*100) if len(cols) else None
    results.append(row)

res = pd.DataFrame(results)
res.to_csv("/home/claude/killzone_results.csv", index=False)

pd.set_option("display.width", 200)
for r in results:
    print(f"\n{r['window']}   (дней: {r['days']}, окно {r['length_min']} мин)")
    def fmt(x):
        return f"{x:5.1f}" if x is not None else "  n/a"
    print(f"   доля дн. экстремумов : {r['extreme_share']*100:5.1f}%   перц. сутки {fmt(r['extreme_share_pct'])} | перц. сессия {fmt(r.get('extreme_share_pct_sess'))}")
    print(f"   ход в ATR            : {r['avg_move_atr']:5.2f}    перц. сутки {fmt(r['avg_move_atr_pct'])} | перц. сессия {fmt(r.get('avg_move_atr_pct_sess'))}")
    print(f"   направленность       : {r['directionality']:5.2f}    перц. сутки {fmt(r['directionality_pct'])} | перц. сессия {fmt(r.get('directionality_pct_sess'))}")
    if r["cont"] is not None:
        print(f"   продолжение          : {r['cont']*100:5.1f}%   перц. сутки {fmt(r['cont_pct'])} | перц. сессия {fmt(r.get('cont_pct_sess'))}")

print("\nsaved -> killzone_results.csv")
