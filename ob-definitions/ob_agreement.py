"""
Насколько расходится разметка order block между разными реализациями.

Берём пять определений, которые ходят в сообществе и в открытом коде,
применяем к одному и тому же участку данных и меряем совпадение.

Ни одно определение здесь не выдумано: все они встречаются в описаниях
индикаторов и обучающих материалах. Расхождения между ними и есть предмет
измерения.
"""
import warnings, io, contextlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

df = pd.read_pickle("/home/claude/es5.pkl")
print("bars:", len(df))

o = df["open"].values.astype(float); h = df["high"].values.astype(float)
l = df["low"].values.astype(float);  c = df["close"].values.astype(float)
v = df["volume"].values.astype(float)
n = len(df)
idx = df.index
sess = idx.normalize()

# ATR(14) на пятиминутках
prev_c = np.concatenate([[c[0]], c[:-1]])
tr = np.maximum(h-l, np.maximum(np.abs(h-prev_c), np.abs(l-prev_c)))
atr = pd.Series(tr).rolling(14).mean().values

def swings(lb):
    """свинговые точки: экстремум среди lb баров слева и справа"""
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(lb, n-lb):
        w_h = h[i-lb:i+lb+1]; w_l = l[i-lb:i+lb+1]
        if h[i] == w_h.max(): sh[i] = True
        if l[i] == w_l.min(): sl[i] = True
    return sh, sl

SH5, SL5 = swings(5)
SH10, SL10 = swings(10)

# ---------------------------------------------------------------
# Пять определений. Каждое возвращает список (bar, lo, hi, dir)
# ---------------------------------------------------------------

def defn_A_basic_impulse():
    """A. Последняя противоположная свеча перед импульсом > 1 ATR.
       Зона = весь диапазон свечи (high-low)."""
    out = []
    for i in range(1, n-3):
        if not np.isfinite(atr[i]) or atr[i] <= 0: continue
        if sess[i] != sess[i+3]: continue
        move = c[i+3] - c[i]
        if move > atr[i] and c[i] < o[i]:
            out.append((i, l[i], h[i], 1))
        elif move < -atr[i] and c[i] > o[i]:
            out.append((i, l[i], h[i], -1))
    return out

def defn_B_body_only():
    """B. То же самое, но зона = только тело свечи (open-close).
       Очень распространённый вариант."""
    out = []
    for i in range(1, n-3):
        if not np.isfinite(atr[i]) or atr[i] <= 0: continue
        if sess[i] != sess[i+3]: continue
        move = c[i+3] - c[i]
        if move > atr[i] and c[i] < o[i]:
            out.append((i, c[i], o[i], 1))
        elif move < -atr[i] and c[i] > o[i]:
            out.append((i, o[i], c[i], -1))
    return out

def defn_C_requires_fvg():
    """C. Последняя противоположная свеча перед движением,
       которое оставило fair value gap."""
    out = []
    for i in range(1, n-3):
        if sess[i] != sess[i+3]: continue
        # ищем FVG в пределах трёх баров после
        has_fvg_up = any(l[j+1] > h[j-1] for j in range(i+1, i+3))
        has_fvg_dn = any(h[j+1] < l[j-1] for j in range(i+1, i+3))
        if has_fvg_up and c[i] < o[i]:
            out.append((i, l[i], h[i], 1))
        elif has_fvg_dn and c[i] > o[i]:
            out.append((i, l[i], h[i], -1))
    return out

def defn_D_requires_bos():
    """D. Требует подтверждения сломом структуры:
       импульс должен пробить ближайшую свинговую точку."""
    out = []
    last_sh = -1; last_sl = -1
    for i in range(1, n-6):
        if SH10[i]: last_sh = i
        if SL10[i]: last_sl = i
        if sess[i] != sess[i+6]: continue
        fwd_hi = h[i+1:i+7].max(); fwd_lo = l[i+1:i+7].min()
        if last_sh > 0 and fwd_hi > h[last_sh] and c[i] < o[i]:
            out.append((i, l[i], h[i], 1))
        elif last_sl > 0 and fwd_lo < l[last_sl] and c[i] > o[i]:
            out.append((i, l[i], h[i], -1))
    return out

def defn_E_highest_volume():
    """E. Свеча с наибольшим объёмом в последних пяти барах перед импульсом."""
    out = []
    for i in range(5, n-3):
        if not np.isfinite(atr[i]) or atr[i] <= 0: continue
        if sess[i-5] != sess[i+3]: continue
        move = c[i+3] - c[i]
        if abs(move) <= atr[i]: continue
        window = np.arange(i-4, i+1)
        k = window[np.argmax(v[i-4:i+1])]
        direction = 1 if move > 0 else -1
        out.append((int(k), l[k], h[k], direction))
    # дедуп
    seen = set(); res = []
    for rec in out:
        if rec[0] in seen: continue
        seen.add(rec[0]); res.append(rec)
    return res

DEFS = {
    "A: свеча перед импульсом, вся свеча": defn_A_basic_impulse,
    "B: то же, но только тело":            defn_B_body_only,
    "C: требует появления FVG":            defn_C_requires_fvg,
    "D: требует слома структуры":          defn_D_requires_bos,
    "E: свеча с максимальным объёмом":     defn_E_highest_volume,
}

print("\n" + "="*78)
print("СКОЛЬКО ЗОН НАХОДИТ КАЖДОЕ ОПРЕДЕЛЕНИЕ")
print("="*78)
res = {}
for name, fn in DEFS.items():
    zones = fn()
    res[name] = zones
    print(f"{name:38s} {len(zones):6d} зон   ({100*len(zones)/n:5.2f}% баров)")

# добавим реальную библиотеку
try:
    with contextlib.redirect_stdout(io.StringIO()):
        from smartmoneyconcepts import smc
    sub = df.reset_index(drop=True)
    sh = smc.swing_highs_lows(sub, swing_length=10)
    ob = smc.ob(sub, sh)
    col = ob.iloc[:,0]
    lib = [(i, l[i], h[i], int(col.iloc[i])) for i in range(n) if not pd.isna(col.iloc[i])]
    res["F: библиотека smartmoneyconcepts"] = lib
    print(f"{'F: библиотека smartmoneyconcepts':38s} {len(lib):6d} зон   ({100*len(lib)/n:5.2f}% баров)")
except Exception as e:
    print("библиотека недоступна:", e)

# ---------------------------------------------------------------
# Совпадение: считаем по барам, помеченным как OB
# ---------------------------------------------------------------
names = list(res.keys())
sets = {k: set(x[0] for x in vlist) for k, vlist in res.items()}

print("\n" + "="*78)
print("ПОПАРНОЕ СОВПАДЕНИЕ РАЗМЕТКИ (коэффициент Жаккара, %)")
print("="*78)
print(" " * 40 + "".join(f"{chr(65+j):>7s}" for j in range(len(names))))
for i, a in enumerate(names):
    row = f"{a[:38]:38s}  "
    for b in names:
        inter = len(sets[a] & sets[b]); union = len(sets[a] | sets[b])
        row += f"{100*inter/union if union else 0:6.1f} "
    print(row)

# согласие всех
allsets = [sets[k] for k in names]
inter_all = set.intersection(*allsets)
union_all = set.union(*allsets)
print(f"\nбаров, помеченных ХОТЯ БЫ одним определением: {len(union_all)}")
print(f"баров, помеченных ВСЕМИ определениями:        {len(inter_all)}")
print(f"доля полного согласия: {100*len(inter_all)/len(union_all):.2f}%")

# сколько определений в среднем помечает бар
from collections import Counter
cnt = Counter()
for s_ in allsets:
    for b in s_:
        cnt[b] += 1
dist = Counter(cnt.values())
print("\nна скольких определениях сошлись помеченные бары:")
for k in sorted(dist):
    print(f"   {k} из {len(names)}: {dist[k]:6d} баров ({100*dist[k]/len(union_all):5.1f}%)")

pd.DataFrame([{"definition": k, "zones": len(vv)} for k, vv in res.items()]).to_csv(
    "/home/claude/ob_definitions_counts.csv", index=False)
print("\nsaved")
