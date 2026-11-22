"""1% long setups: oversold hook at support, with room left in the day's predicted range.

Long when ALL of these hold on a fast candle (15m by default):
  1. RSI(6) was below RSI_LO and turns up (the "hook").
  2. Room: the day's predicted high (daily-range model, median) is at least
     ROOM above price, so a +1% move fits inside the expected day.
  3. Not breaking down: price is above the day's predicted low (10% quantile).
  4. Support: price is within SUPPORT_BAND above at least one support from the
     slower time frame (1h): strongest liquidity cluster below, MA99, MA200,
     the last swing low, or the day's predicted low (median).
  5. (variant) Trend: slow close above its MA25.
Exit: TP +1%, SL -1%, or after MAX_HOLD bars at market. Fees 0.2% round trip.

backtest(): compares the rule set with random entries using the same exits.
signals(): live check for the latest candle (used by the 15-minute workflow).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import liquidity as L
from .paths import DATA, ROOT

RSI_LO = 20
ROOM = 0.012
SUPPORT_BAND = 0.006
TP, SL = 0.01, 0.01
FEE = 0.002
COINS = {"BTC": "btc", "ETH": "eth", "SOL": "sol"}
RNG = np.random.default_rng(11)


def rsi(c, n):
    x = c.diff()
    g = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    lo = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / lo)


def slow_context(slow):
    """Per slow bar: MA25/99/200, last confirmed swing low, strongest liquidity cluster below."""
    c = slow.close
    ctx = pd.DataFrame(index=slow.t)
    ctx["s_close"] = c.to_numpy()
    for n in (25, 99, 200):
        ctx[f"ma{n}"] = c.rolling(n).mean().to_numpy()
    lows = slow.low.to_numpy()
    sw = np.full(len(slow), np.nan)
    last = np.nan
    p = L.PIVOT
    for i in range(len(slow)):
        j = i - p
        if j >= p and lows[j] == lows[j - p:i + 1].min():
            last = lows[j]
        sw[i] = last
    ctx["swing_low"] = sw
    liq = np.full(len(slow), np.nan)

    def cb(i, eng, swept):
        px, w, _, _ = eng.weights(i)
        if not len(px):
            return
        price = slow.close.iat[i]
        m = (px < price * (1 - 0.003)) & (px > price * (1 - 0.05))
        if m.any():
            step = price * 0.0025
            b = np.floor((px[m] - price) / step).astype(int)
            s = pd.Series(w[m]).groupby(b).sum()
            liq[i] = price + (s.idxmax() + 0.5) * step

    L.run_engine(slow.reset_index(drop=True), cb)
    ctx["liq_below"] = liq
    return ctx


def daily_preds(key):
    """Day -> predicted high/low prices for that UTC day (from the daily-range model)."""
    p = DATA / f"daily_bt_{key}.csv"
    out = {}
    if p.exists():
        b = pd.read_csv(p, parse_dates=["target_date"])
        b = b[b.k == 1]
        for r in b.itertuples():
            out[r.target_date.date()] = {q: r.close * np.exp(getattr(r, q) * r.scale)
                                         for q in ("hi_q50", "lo_q10", "lo_q50")}
    lat = DATA / "daily_range_latest.json"
    if lat.exists():
        for r in json.loads(lat.read_text()).get(key, []):
            out[pd.Timestamp(r["target_date"]).date()] = {q: r[q] for q in ("hi_q50", "lo_q10", "lo_q50")}
    return out


def build(coin, fast_tf="15m", slow_tf="1h"):
    key = COINS[coin]
    fast = L.load(coin, fast_tf)
    slow = L.load(coin, slow_tf)
    ctx = slow_context(slow)
    # a fast bar may only use slow bars that had CLOSED before it closed
    step = pd.Timedelta(slow_tf.replace("m", "min"))
    ctx.index = ctx.index + step               # time the slow bar closes
    f = fast.copy()
    f["close_t"] = f.t + pd.Timedelta(fast_tf.replace("m", "min"))
    f = pd.merge_asof(f.sort_values("close_t"), ctx.sort_index(), left_on="close_t", right_index=True)
    f["rsi6"] = rsi(f.close, 6)
    dp = daily_preds(key)
    day = f.t.dt.date
    for q in ("hi_q50", "lo_q10", "lo_q50"):
        f[q] = [dp.get(d, {}).get(q, np.nan) for d in day]
    return f.reset_index(drop=True)


def conditions(f):
    c = f.close
    hook = (f.rsi6.shift(1) < RSI_LO) & (f.rsi6 > f.rsi6.shift(1))
    room = (f.hi_q50 / c - 1) >= ROOM
    hold = c >= f.lo_q10
    sup = pd.Series(False, index=f.index)
    for col in ("liq_below", "ma99", "ma200", "swing_low", "lo_q50"):
        s = f[col]
        sup |= (s <= c) & (c / s - 1 <= SUPPORT_BAND)
    trend = f.s_close > f.ma25
    return {"hook": hook, "room": room, "hold": hold, "support": sup, "trend": trend}


def simulate(f, entries, max_hold):
    h, lo, c = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    trades, busy_until = [], -1
    for i in np.flatnonzero(entries):
        if i <= busy_until or i + 1 >= len(c):
            continue
        e = c[i]
        res, j_exit = None, None
        for j in range(i + 1, min(i + 1 + max_hold, len(c))):
            hit_sl, hit_tp = lo[j] <= e * (1 - SL), h[j] >= e * (1 + TP)
            if hit_sl:                     # same-bar ambiguity counts as a loss
                res, j_exit = -SL, j
                break
            if hit_tp:
                res, j_exit = TP, j
                break
        if res is None:
            j_exit = min(i + max_hold, len(c) - 1)
            res = c[j_exit] / e - 1
        trades.append({"i": i, "t": str(f.t.iat[i]), "entry": e, "ret": res - FEE, "bars": j_exit - i})
        busy_until = j_exit
    return pd.DataFrame(trades)


def stats(t, days):
    if t.empty:
        return {"n": 0}
    eq = t.ret.cumsum()
    return {"n": int(len(t)), "per_day": round(len(t) / days, 2), "win": round(float((t.ret > 0).mean()) * 100, 1),
            "avg_net_pct": round(float(t.ret.mean()) * 100, 3), "total_net_pct": round(float(t.ret.sum()) * 100, 2),
            "max_dd_pct": round(float((eq.cummax() - eq).max()) * 100, 2), "avg_hold_bars": round(float(t.bars.mean()), 1)}


def backtest(fast_tf="15m", slow_tf="1h", max_hold=96):
    out = {}
    pooled = {}
    for coin in COINS:
        f = build(coin, fast_tf, slow_tf)
        f = f[f.hi_q50.notna() & f.ma200.notna()].reset_index(drop=True)
        days = max(1, (f.t.iat[-1] - f.t.iat[0]).days)
        cd = conditions(f)
        variants = {
            "1_rsi_hook": cd["hook"],
            "2_hook+room": cd["hook"] & cd["room"] & cd["hold"],
            "3_hook+room+support": cd["hook"] & cd["room"] & cd["hold"] & cd["support"],
            "4_+trend": cd["hook"] & cd["room"] & cd["hold"] & cd["support"] & cd["trend"],
        }
        res = {"period": [str(f.t.iat[0])[:16], str(f.t.iat[-1])[:16]], "days": days}
        for name, e in variants.items():
            t = simulate(f, e.to_numpy(), max_hold)
            res[name] = stats(t, days)
            pooled.setdefault(name, []).append(t)
        # random entries, same exits, 200 repetitions
        n = max(res["3_hook+room+support"].get("n", 0), 20)
        rs = []
        for _ in range(200):
            e = np.zeros(len(f), bool)
            e[RNG.choice(len(f) - max_hold, size=min(n * 3, len(f) - max_hold), replace=False)] = True
            rs.append(simulate(f, e, max_hold))
        r = pd.concat(rs)
        res["random"] = {"win": round(float((r.ret > 0).mean()) * 100, 1), "avg_net_pct": round(float(r.ret.mean()) * 100, 3)}
        pooled.setdefault("random", []).append(r)
        out[coin] = res
        print(coin, fast_tf, json.dumps(res))
    pool = {}
    for name, ts in pooled.items():
        t = pd.concat(ts)
        pool[name] = {"n": int(len(t)), "win": round(float((t.ret > 0).mean()) * 100, 1),
                      "avg_net_pct": round(float(t.ret.mean()) * 100, 3)}
    out["pooled"] = pool
    print("pooled", json.dumps(pool))
    return out


# ── live ───────────────────────────────────────────────────────────────
def signals(fast_tf="15m", slow_tf="1h"):
    """Evaluate the latest CLOSED fast candle for each coin."""
    found, watch = [], []
    for coin in COINS:
        f = build(coin, fast_tf, slow_tf)
        f = f.iloc[:-1]                       # last row is still forming
        cd = conditions(f)
        i = len(f) - 1
        r = f.iloc[i]
        ok = {k: bool(v.iat[i]) for k, v in cd.items()}
        supports = {col: float(r[col]) for col in ("liq_below", "ma99", "ma200", "swing_low", "lo_q50")
                    if pd.notna(r[col]) and r[col] <= r.close and r.close / r[col] - 1 <= SUPPORT_BAND}
        info = {"coin": coin, "time": str(r.t)[:16], "price": float(r.close), "rsi6": round(float(r.rsi6), 1),
                "pred_hi": float(r.hi_q50) if pd.notna(r.hi_q50) else None,
                "pred_lo10": float(r.lo_q10) if pd.notna(r.lo_q10) else None,
                "supports": supports, "ok": ok}
        if ok["hook"] and ok["room"] and ok["hold"] and ok["support"]:
            e = float(r.close)
            info.update(entry=e, tp=e * (1 + TP), sl=e * (1 - SL), strong=ok["trend"])
            found.append(info)
        elif r.rsi6 < RSI_LO + 5:
            watch.append(info)
    return found, watch


REPORT = DATA / "scalp_backtest.json"
GATE_RULE = "3_hook+room+support"


def gate():
    """Alerts go live only when the setup has beaten random entries AFTER fees on recent data."""
    if not REPORT.exists():
        return False, "no backtest yet"
    p = json.loads(REPORT.read_text())["15m"]["pooled"]
    rule, rnd = p.get(GATE_RULE, {}), p.get("random", {})
    n, avg, ravg = rule.get("n", 0), rule.get("avg_net_pct", -9), rnd.get("avg_net_pct", 0)
    ok = n >= 40 and avg > 0.05 and avg > ravg + 0.10
    why = (f"{n} trades, {rule.get('win', 0)}% wins, {avg:+.2f}% per trade after fees "
           f"(random entries {ravg:+.2f}%)")
    return ok, why


def fmtp(p):
    return f"${p:,.0f}" if p >= 1000 else f"${p:,.2f}" if p >= 10 else f"${p:.4f}"


def live():
    from .alerts import send
    ok, why = gate()
    found, watch = signals()
    for x in found:
        sup = ", ".join(f"{k.replace('_', ' ')} {fmtp(v)}" for k, v in x["supports"].items())
        msg = (f"{'🟢' if ok else '🔍 (paper)'} {x['coin']} 1% long setup ({x['time']} UTC, 15m)\n"
               f"Entry {fmtp(x['entry'])} · TP {fmtp(x['tp'])} · SL {fmtp(x['sl'])}\n"
               f"RSI6 {x['rsi6']} hooking up · day's predicted high {fmtp(x['pred_hi'])} · support: {sup}"
               + (" · 1h trend up" if x["strong"] else " · against the 1h trend") + "\n"
               + ("Backtest: " + why if ok else "PAPER ONLY — the backtest has not beaten random entries yet: " + why))
        print(msg)
        if ok:
            send(msg)
    if not found:
        print("no setup;", "watching:", [(w["coin"], w["rsi6"]) for w in watch], "| gate:", ok, why)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        rep = {"made": datetime.now(timezone.utc).isoformat(),
               "15m": backtest("15m", "1h", 96), "1h": backtest("1h", "4h", 24)}
        REPORT.write_text(json.dumps(rep, indent=1))
    else:
        live()
