import warnings, io, contextlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import importlib.util

with contextlib.redirect_stdout(io.StringIO()):
    from smartmoneyconcepts import smc

spec = importlib.util.spec_from_file_location("la2", "/home/claude/lookahead_audit2.py")
la2 = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(la2)

def synth(n, seed):
    rng = np.random.RandomState(seed)
    sigma = 0.0009
    ret = rng.normal(0, sigma, n)
    close = 20000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[20000.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma*0.6, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma*0.6, n)))
    vol = rng.randint(500, 5000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=idx)

# ---------- experiment 1: settling time across seeds ----------
print("=== SETTLING TIME (bars of future needed before label is final) ===")
rows = []
for seed in [11, 22, 33]:
    df = synth(2200, seed)
    for which, sl, maxf in [("fvg",50,20), ("swing",10,40), ("swing",25,80), ("swing",50,120)]:
        r = la2.run_audit(which, df, 700, 900, maxf, sl)
        s = r["settle"]
        if len(s)==0:
            continue
        rows.append({
            "seed": seed, "function": which, "swing_length": sl,
            "events": int(r["final_events"]),
            "pct_late": round(100.0*(s>0).sum()/len(s),1),
            "median_future": float(np.median(s)),
            "max_future": int(s.max()),
        })
        print(f"seed {seed} | {which:6s} sl={sl:3d} | events {r['final_events']:4d} | "
              f"late {rows[-1]['pct_late']:5.1f}% | median {np.median(s):5.0f} bars")

res1 = pd.DataFrame(rows)
res1.to_csv("/home/claude/audit_settling.csv", index=False)

# ---------- experiment 2: live repaint of swing labels ----------
print("\n=== LIVE REPAINT: is the freshly closed bar marked as a swing? ===")
rep_rows = []
for seed in [11,22,33]:
    df = synth(1600, seed)
    for sl in [10, 25, 50]:
        sh_full = smc.swing_highs_lows(df, swing_length=sl)
        labeled_live = 0; survived = 0; trials = 0
        for L in range(900, 1000):
            sub = df.iloc[:L]
            v = smc.swing_highs_lows(sub, swing_length=sl).iloc[-1,0]
            trials += 1
            if not (isinstance(v,float) and np.isnan(v)):
                labeled_live += 1
                vf = sh_full.iloc[L-1,0]
                if not (isinstance(vf,float) and np.isnan(vf)) and vf == v:
                    survived += 1
        rep_rows.append({"seed":seed,"swing_length":sl,"bars":trials,
                         "labeled_live":labeled_live,"survived":survived,
                         "vanished_pct": round(100.0*(labeled_live-survived)/max(labeled_live,1),1)})
        print(f"seed {seed} | sl={sl:3d} | live signals {labeled_live:3d}/{trials} | "
              f"survived {survived:3d} | vanished {rep_rows[-1]['vanished_pct']:5.1f}%")

res2 = pd.DataFrame(rep_rows)
res2.to_csv("/home/claude/audit_repaint.csv", index=False)

# ---------- experiment 3: liquidity threshold depends on全 sample ----------
print("\n=== LIQUIDITY: does the zone threshold depend on how much history you load? ===")
df = synth(3000, 11)
liq_rows = []
for L in [1000, 1500, 2000, 2500, 3000]:
    sub = df.iloc[:L]
    pip_range = (sub["high"].max() - sub["low"].min()) * 0.01
    sh = smc.swing_highs_lows(sub, swing_length=50)
    lq = smc.liquidity(sub, sh)
    n_zones = int(lq.iloc[:,0].notna().sum())
    # zones detected within the first 1000 bars only
    n_zones_first1000 = int(lq.iloc[:1000,0].notna().sum())
    liq_rows.append({"history_bars":L,"pip_range":round(pip_range,2),
                     "zones_total":n_zones,"zones_in_first_1000":n_zones_first1000})
    print(f"history {L:5d} bars | threshold {pip_range:8.2f} | "
          f"zones in first 1000 bars: {n_zones_first1000}")

res3 = pd.DataFrame(liq_rows)
res3.to_csv("/home/claude/audit_liquidity.csv", index=False)
print("\nsaved 3 csv files")
