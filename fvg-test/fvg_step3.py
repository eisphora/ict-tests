import numpy as np, pandas as pd

RNG = np.random.RandomState(101)
real = pd.read_pickle("/home/claude/fvg_real.pkl")
ctrl = pd.read_pickle("/home/claude/fvg_ctrl.pkl")

# ---------- block bootstrap by session ----------
sessions = np.array(sorted(set(real["session"]) | set(ctrl["session"])))
r_groups = {s: g for s, g in real.groupby("session")}
c_groups = {s: g for s, g in ctrl.groupby("session")}

def boot_diff(metric, H, B=1000):
    diffs = np.empty(B)
    for b in range(B):
        pick = RNG.choice(sessions, size=len(sessions), replace=True)
        rs, cs = [], []
        for s in pick:
            g = r_groups.get(s)
            if g is not None:
                rs.append(g[f"{metric}_{H}"].values)
            g2 = c_groups.get(s)
            if g2 is not None:
                cs.append(g2[f"{metric}_{H}"].values)
        if not rs or not cs:
            diffs[b] = np.nan; continue
        diffs[b] = np.concatenate(rs).mean() - np.concatenate(cs).mean()
    return np.nanpercentile(diffs, [2.5, 97.5]) * 100

print("="*74)
print("DIFFERENCE (FVG minus matched random), percentage points, 95% block bootstrap")
print("="*74)
print(f"{'horizon':>9} | {'metric':>6} | {'FVG':>6} | {'random':>7} | {'diff':>6} | {'95% CI':>18}")
print("-"*74)
for H in [5, 20, 60, "session"]:
    for metric in ["near", "full"]:
        a = real[f"{metric}_{H}"].mean()*100
        b = ctrl[f"{metric}_{H}"].mean()*100
        lo, hi = boot_diff(metric, H, B=400)
        star = "" if (lo < 0 < hi) else "  <-- significant"
        print(f"{str(H):>9} | {metric:>6} | {a:5.1f}% | {b:6.1f}% | {a-b:+5.1f} | [{lo:+5.2f}, {hi:+5.2f}]{star}")
    print("-"*74)

# ---------- slice 1: fill curve by size bucket ----------
print("\n" + "="*74)
print("PRE-SPECIFIED SLICE 1: full fill within 60 bars, by zone size")
print("="*74)
bins = [0, 0.3, 0.7, 1.2, 99]
names = ["<0.3 ATR", "0.3-0.7", "0.7-1.2", ">1.2 ATR"]
real["bucket"] = pd.cut(real["height_atr"], bins, labels=names)
ctrl["bucket"] = pd.cut(ctrl["height_atr"], bins, labels=names)
print(f"{'bucket':>10} | {'n':>7} | {'FVG':>7} | {'random':>7} | {'diff':>6}")
print("-"*74)
for nm in names:
    r = real[real["bucket"] == nm]; c = ctrl[ctrl["bucket"] == nm]
    if len(r) == 0: continue
    a = r["full_60"].mean()*100; b = c["full_60"].mean()*100
    print(f"{nm:>10} | {len(r):7d} | {a:6.1f}% | {b:6.1f}% | {a-b:+5.1f}")

# ---------- slice 2: threshold sensitivity ----------
print("\n" + "="*74)
print("PRE-SPECIFIED SLICE 2: how many FVG exist, by minimum size threshold")
print("="*74)
total_bars = 111533
for thr in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
    k = (real["height_atr"] >= thr).sum()
    print(f"  min size {thr:>4} ATR -> {k:6d} zones   ({100*k/total_bars:5.2f}% of bars)")

# ---------- slice 3: direction ----------
print("\n" + "="*74)
print("PRE-SPECIFIED SLICE 3: by direction (full fill, 60 bars)")
print("="*74)
for d, nm in [(1, "bullish"), (-1, "bearish")]:
    r = real[real["dir"] == d]; c = ctrl[ctrl["dir"] == d]
    print(f"  {nm:>8}: n={len(r):6d}  FVG {r['full_60'].mean()*100:5.1f}%  random {c['full_60'].mean()*100:5.1f}%  diff {(r['full_60'].mean()-c['full_60'].mean())*100:+5.1f}")

# ---------- slice 4: by hour ----------
print("\n" + "="*74)
print("PRE-SPECIFIED SLICE 4: by hour of day (full fill, 60 bars)")
print("="*74)
for hh in sorted(real["hour"].unique()):
    r = real[real["hour"] == hh]; c = ctrl[ctrl["hour"] == hh]
    if len(r) < 100: continue
    print(f"  {hh:02d}:00  n={len(r):5d}  FVG {r['full_60'].mean()*100:5.1f}%  random {c['full_60'].mean()*100:5.1f}%  diff {(r['full_60'].mean()-c['full_60'].mean())*100:+5.1f}")
