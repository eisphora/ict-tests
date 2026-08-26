"""
Look-ahead / repaint audit for SMC libraries.

For each bar i we ask two separate questions:

  (1) SETTLING TIME. On the final (full-history) run bar i carries label L.
      What is the minimal number of future bars f such that every prefix of
      length >= i+1+f already assigns L to bar i?
      f = 0  -> the label was correct the moment bar i closed (causal).
      f > 0  -> the label could not have been known in real time.

  (2) REPAINT. Was bar i labelled as an event at the moment it closed, and
      that label later disappeared or changed? Those are signals a live trader
      would have acted on and that vanish from the backtest.

The test runs on synthetic GBM bars: it probes the code, not the market.
"""
import warnings, io, contextlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    from smartmoneyconcepts import smc

RNG = np.random.RandomState(20260821)


def synth_ohlc(n, s0=20000.0, sigma=0.0009):
    ret = RNG.normal(0, sigma, n)
    close = s0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[s0], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(RNG.normal(0, sigma * 0.6, n)))
    low = np.minimum(open_, close) * (1 - np.abs(RNG.normal(0, sigma * 0.6, n)))
    vol = RNG.randint(500, 5000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def get_labels(df, which, swing_length):
    """1-D label per bar; NaN where no event."""
    if which == "fvg":
        return smc.fvg(df).iloc[:, 0]
    sh = smc.swing_highs_lows(df, swing_length=swing_length)
    if which == "swing":
        return sh.iloc[:, 0]
    if which == "bos_choch":
        r = smc.bos_choch(df, sh)
        a, b = r.iloc[:, 0], r.iloc[:, 1]
        out = a.fillna(0) * 10 + b.fillna(0)
        return out.replace(0, np.nan)
    if which == "ob":
        return smc.ob(df, sh).iloc[:, 0]
    if which == "liquidity":
        return smc.liquidity(df, sh).iloc[:, 0]
    raise ValueError(which)


def norm(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return float(v)


def run_audit(which, df, probe_lo, probe_hi, max_future, swing_length):
    """Collect the label of every probe bar across growing prefixes."""
    need_len = probe_hi + max_future + 1
    assert need_len <= len(df)

    # table[i][f] = label of bar i on the prefix that ends f bars after i
    table = {i: [None] * (max_future + 1) for i in range(probe_lo, probe_hi)}
    for L in range(probe_lo + 1, need_len + 1):
        try:
            lab = get_labels(df.iloc[:L], which, swing_length)
        except Exception:
            continue
        for i in range(probe_lo, min(probe_hi, L)):
            f = L - i - 1
            if 0 <= f <= max_future:
                table[i][f] = norm(lab.iloc[i])

    settle, repaints, final_events, at_close_events = [], 0, 0, 0
    for i, vals in table.items():
        final = vals[max_future]
        at_close = vals[0]
        if at_close is not None:
            at_close_events += 1
        if final is not None:
            final_events += 1
            f_need = 0
            for f in range(max_future, -1, -1):
                if vals[f] != final:
                    f_need = f + 1
                    break
            settle.append(f_need)
        if at_close is not None and at_close != final:
            repaints += 1

    return {
        "settle": np.array(settle, dtype=float),
        "repaints": repaints,
        "final_events": final_events,
        "at_close_events": at_close_events,
        "probe_bars": probe_hi - probe_lo,
        "max_future": max_future,
    }


def report(which, r, swing_length):
    s = r["settle"]
    if len(s) == 0:
        return None
    late = (s > 0).sum()
    return {
        "function": which,
        "swing_length": swing_length,
        "events_found": int(r["final_events"]),
        "events_not_known_at_close": int(late),
        "pct_not_known_at_close": round(100.0 * late / len(s), 1),
        "median_future_bars": float(np.median(s)),
        "p90_future_bars": float(np.percentile(s, 90)),
        "max_future_bars": int(s.max()),
        "repainted_signals": int(r["repaints"]),
        "signals_shown_live": int(r["at_close_events"]),
    }


if __name__ == "__main__":
    N = 3000
    df = synth_ohlc(N)
    MAXF = 60
    LO, HI = 800, 1400          # wide probe window, plenty of events

    rows = []
    configs = [
        ("fvg", 50),
        ("swing", 50),
        ("swing", 10),
        ("bos_choch", 50),
        ("ob", 50),
        ("liquidity", 50),
    ]
    for which, sl in configs:
        r = run_audit(which, df, LO, HI, MAXF, sl)
        rep = report(which, r, sl)
        if rep is None:
            print(f"{which:12s} sl={sl:3d}  no events found in probe window")
            continue
        rows.append(rep)
        print(f"{which:12s} sl={sl:3d}  events {rep['events_found']:4d}  "
              f"not-known-at-close {rep['pct_not_known_at_close']:5.1f}%  "
              f"median {rep['median_future_bars']:4.0f}  max {rep['max_future_bars']:3d}  "
              f"repaints {rep['repainted_signals']:4d}")

    out = pd.DataFrame(rows)
    out.to_csv("/home/claude/lookahead_results.csv", index=False)
    print("\nsaved -> lookahead_results.csv")
