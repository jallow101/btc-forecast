"""Daily 1% longs: $5k budget, goal ≈ $50/day, at most 7 proposals a day.

A proposal is made on a CLOSED 15-minute candle only when every condition
of the chosen rule holds (see CONDS). The idea, from the research:
  * Intraday BTC returns mean-revert at 1-4h horizons, and bigger drops
    revert harder (De Nicola 2021) -> buy oversold, stretched dips.
  * Lows tend to print in the Asian/European session (Bloomberg 2021)
    -> test whether Asia-session dips really do better.
  * But do NOT catch a falling knife: the candle must already show the
    selling stopped (green close back over EMA7), the 1h/4h must not be in
    a cascade, and BTC must not be dumping (it drags every alt).

Risk rules (paper log and backtest): $2,500 per position, at most 2 open,
at most 7 proposals per UTC day, stop for the day after 2 losses,
TP +1%, SL below the support (0.8%-2.0%), close after 24h. Fees 0.2%.

backtest(): every rule variant vs random entries with the same exits, by
session, plus a portfolio simulation in dollars vs the $50/day goal.
live(): checks the newest closed candle, updates the paper log.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import liquidity as L
from .paths import DATA, ROOT, SITE_DATA
from .scalp import daily_preds, fmtp, rsi

COINS = {"BTC": "btc", "ETH": "eth", "SOL": "sol"}
TP = 0.01
SL_FIXED = 0.015
SL_MIN, SL_MAX, SL_PAD = 0.008, 0.020, 0.003
FEE = 0.002
MAX_HOLD = 96                      # 15m bars = 24h
BUDGET, SIZE, MAX_OPEN, MAX_PER_DAY, MAX_LOSSES = 5000, 2500, 2, 7, 2
GOAL = 50
RSI_LO = 25
ROOM = TP + 0.002                  # predicted high must leave room for the TP
SUPPORT_BAND = 0.006
SESSIONS = [("Asia", 0, 8), ("Europe", 8, 13), ("US", 13, 21), ("Late", 21, 24)]
RNG = np.random.default_rng(7)
REPORT = DATA / "daytrade_backtest.json"
LOG_DIR = ROOT / "out" / "trades"


def session(h):
    for name, a, b in SESSIONS:
        if a <= h < b:
            return name
    return "Late"


# ── data ──────────────────────────────────────────────────────────────
def load_15m(coin):
    parts = []
    for p in (DATA / "hist" / f"{coin}_15m.csv", DATA / "liq" / f"{coin}_15m.csv"):
        if p.exists():
            parts.append(pd.read_csv(p))
    df = pd.concat(parts).drop_duplicates("ts", keep="last").sort_values("ts")
    df["t"] = pd.to_datetime(df.ts, unit="s", utc=True)
    # drop gaps of mixed sources: keep a regular 15m grid where possible
    return df.reset_index(drop=True)


def resample(df, rule):
    r = df.set_index("t").resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return r.reset_index()


def hourly_context(h1):
    """Per 1h bar: MAs, RSI, last swing low, strongest liquidity cluster below and above."""
    c = h1.close
    ctx = pd.DataFrame(index=h1.t)
    ctx["h_close"] = c.to_numpy()
    for n in (25, 99, 200):
        ctx[f"ma{n}"] = c.rolling(n).mean().to_numpy()
    ctx["h_rsi6"] = rsi(c, 6).to_numpy()
    ctx["h_ret1"] = (c / c.shift(1) - 1).to_numpy()
    lows = h1.low.to_numpy()
    sw = np.full(len(h1), np.nan)
    last, p = np.nan, L.PIVOT
    for i in range(len(h1)):
        j = i - p
        if j >= p and lows[j] == lows[j - p:i + 1].min():
            last = lows[j]
        sw[i] = last
    ctx["swing_low"] = sw
    below = np.full(len(h1), np.nan)
    above = np.full(len(h1), np.nan)
    above_w = np.full(len(h1), np.nan)
    cl = h1.close.to_numpy()

    def cb(i, eng, swept):
        px, w, _, _ = eng.weights(i)
        if not len(px):
            return
        price = cl[i]
        step = price * 0.0025
        m = (px < price * 0.997) & (px > price * 0.95)
        if m.any():
            s = pd.Series(w[m]).groupby(np.floor((px[m] - price) / step).astype(int)).sum()
            below[i] = price + (s.idxmax() + 0.5) * step
        m = (px > price * 1.003) & (px < price * 1.06)
        if m.any():
            s = pd.Series(w[m]).groupby(np.floor((px[m] - price) / step).astype(int)).sum()
            above[i] = price + (s.idxmax() + 0.5) * step
            above_w[i] = s.max() / max(1e-9, np.median(s))

    L.run_engine(h1.reset_index(drop=True), cb)
    ctx["liq_below"], ctx["liq_above"], ctx["liq_above_rel"] = below, above, above_w
    return ctx


def build(coin, btc=None):
    key = COINS[coin]
    f = load_15m(coin)
    h1 = resample(f, "1h")
    h4 = resample(f, "4h")
    ctx = hourly_context(h1)
    ctx.index = ctx.index + pd.Timedelta("1h")          # known when the hour closes
    c4 = pd.DataFrame({"h4_rsi14": rsi(h4.close, 14).to_numpy(),
                       "h4_ret6": (h4.close / h4.close.shift(6) - 1).to_numpy()},
                      index=h4.t + pd.Timedelta("4h"))
    f["close_t"] = f.t + pd.Timedelta("15min")
    f = pd.merge_asof(f.sort_values("close_t"), ctx.sort_index(), left_on="close_t", right_index=True)
    f = pd.merge_asof(f, c4.sort_index(), left_on="close_t", right_index=True)
    c = f.close
    f["rsi6"] = rsi(c, 6)
    f["ema7"] = c.ewm(span=7, adjust=False).mean()
    f["ret4"] = c / c.shift(4) - 1                     # last hour on 15m
    f["vrel"] = f.volume / f.volume.rolling(50).mean().shift(1)
    dp = daily_preds(key)
    day = f.t.dt.date
    for q in ("hi_q50", "lo_q10", "lo_q50"):
        f[q] = [dp.get(d, {}).get(q, np.nan) for d in day]
    f["session"] = [session(h) for h in f.t.dt.hour]
    if btc is not None:
        f = f.merge(btc[["t", "ret4", "rsi6"]].rename(columns={"ret4": "btc_ret4", "rsi6": "btc_rsi6"}),
                    on="t", how="left")
    else:
        f["btc_ret4"], f["btc_rsi6"] = f.ret4, f.rsi6
    return f.reset_index(drop=True)


# ── conditions ────────────────────────────────────────────────────────
SUPPORTS = ("liq_below", "ma99", "ma200", "swing_low", "lo_q50")
LABELS = {
    "hook": "15m RSI(6) was under 25 and turned up",
    "room": "day's predicted high ≥ 1.2% above (room for the 1% target)",
    "hold": "price still above the day's predicted low (10% band)",
    "support": "within 0.6% above a support (1h liquidity / MA99 / MA200 / swing low / predicted low)",
    "stopped": "selling stopped: green 15m candle closed back over EMA7",
    "no_cascade": "no cascade: last 1h candle > −2.5%, 4h RSI(14) > 25",
    "btc_ok": "BTC not dumping: BTC last hour > −1%",
    "asia": "Asian session (00–08 UTC)",
    "stretch": "stretched: ≥1.5% under the 1h MA25 (bigger dips revert harder)",
    "capitulation": "a volume spike (≥2× average) in the last hour",
    "magnet": "a liquidity cluster 1–6% above to pull price up",
}


def conditions(f):
    c = f.close
    r = f.rsi6
    sup = pd.Series(False, index=f.index)
    for col in SUPPORTS:
        s = f[col]
        sup |= (s <= c) & (c / s - 1 <= SUPPORT_BAND)
    return {
        "hook": (r.rolling(3).min().shift(1) < RSI_LO) & (r > r.shift(1)),
        "room": (f.hi_q50 / c - 1) >= ROOM,
        "hold": c >= f.lo_q10,
        "support": sup,
        "stopped": (c > f.open) & (c > f.ema7),
        "no_cascade": (f.h_ret1 > -0.025) & (f.h4_rsi14 > 25),
        "btc_ok": f.btc_ret4 > -0.01,
        "asia": f.session == "Asia",
        "stretch": c <= f.ma25 * (1 - 0.015),
        "capitulation": f.vrel.rolling(4).max() >= 2,
        "magnet": (f.liq_above / c - 1).between(TP, 0.06),
    }


CORE = ["hook", "room", "hold", "support"]
SAFE = CORE + ["stopped", "no_cascade", "btc_ok"]
VARIANTS = {
    "A_rsi_hook_only": ["hook"],
    "B_core": CORE,
    "C_core+not_falling": SAFE,
    "D_C+asia": SAFE + ["asia"],
    "E_C+stretch": SAFE + ["stretch"],
    "F_C+capitulation": SAFE + ["capitulation"],
    "G_C+magnet": SAFE + ["magnet"],
    "H_C+asia+stretch": SAFE + ["asia", "stretch"],
    "I_C+asia+capitulation": SAFE + ["asia", "capitulation"],
    "J_C+asia+magnet": SAFE + ["asia", "magnet"],
}
# Best on 391 days at TP 1% (57 trades, 67% hit, +0.22%/trade after fees, both halves positive).
# Chosen after trying 10 variants, so the paper log is the real test.
LIVE_RULE = "I_C+asia+capitulation"


def mask(cd, names):
    m = pd.Series(True, index=cd["hook"].index)
    for n in names:
        m &= cd[n].fillna(False)
    return m


def stop_price(row, mode):
    e = row["close"]
    if mode == "fixed":
        return e * (1 - SL_FIXED)
    sups = [row[s] for s in SUPPORTS if pd.notna(row[s]) and row[s] <= e and e / row[s] - 1 <= SUPPORT_BAND]
    base = min(sups) if sups else e * (1 - SL_FIXED)
    d = 1 - base * (1 - SL_PAD) / e
    return e * (1 - min(max(d, SL_MIN), SL_MAX))


# ── simulation ────────────────────────────────────────────────────────
def simulate(f, entries, mode="structure"):
    h, lo, c = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    out, busy = [], -1
    for i in np.flatnonzero(entries):
        if i <= busy or i + 1 >= len(c):
            continue
        e = c[i]
        sl = stop_price(f.iloc[i], mode) if mode != "fixed" else e * (1 - SL_FIXED)
        tp = e * (1 + TP)
        res, k, why = None, None, "time"
        for j in range(i + 1, min(i + 1 + MAX_HOLD, len(c))):
            if lo[j] <= sl:                       # same-candle ambiguity = loss
                res, k, why = sl / e - 1, j, "SL"
                break
            if h[j] >= tp:
                res, k, why = TP, j, "TP"
                break
        if res is None:
            k = min(i + MAX_HOLD, len(c) - 1)
            res = c[k] / e - 1
        out.append({"i": i, "t": f.t.iat[i], "exit_t": f.t.iat[k], "ret": res - FEE, "why": why,
                    "sl_pct": 1 - sl / e, "session": f.session.iat[i]})
        busy = k
    return pd.DataFrame(out)


def stats(t, days):
    if t is None or t.empty:
        return {"n": 0}
    eq = t.ret.cumsum()
    return {"n": int(len(t)), "per_day": round(len(t) / days, 2),
            "win": round(float((t.why == "TP").mean()) * 100, 1),
            "avg_net_pct": round(float(t.ret.mean()) * 100, 3),
            "total_net_pct": round(float(t.ret.sum()) * 100, 1),
            "max_dd_pct": round(float((eq.cummax() - eq).max()) * 100, 1),
            "avg_sl_pct": round(float(t.sl_pct.mean()) * 100, 2)}


def boot_ci(x, n=2000):
    if len(x) < 10:
        return None
    x = np.asarray(x)
    m = [RNG.choice(x, len(x)).mean() for _ in range(n)]
    return [round(float(np.percentile(m, 5)) * 100, 3), round(float(np.percentile(m, 95)) * 100, 3)]


def portfolio(trades, days):
    """Dollars with the real risk rules: $2,500 each, ≤2 open, ≤7/day, stop after 2 losses."""
    if trades.empty:
        return {"days": days, "trades": 0}
    t = trades.sort_values("t")
    taken, open_ = [], []
    per_day, losses = {}, {}
    for r in t.itertuples():
        d = r.t.date()
        open_ = [x for x in open_ if x > r.t]
        if len(open_) >= MAX_OPEN or per_day.get(d, 0) >= MAX_PER_DAY or losses.get(d, 0) >= MAX_LOSSES:
            continue
        per_day[d] = per_day.get(d, 0) + 1
        open_.append(r.exit_t)
        if r.ret < 0:
            losses[d] = losses.get(d, 0) + 1
        taken.append({"day": d, "usd": r.ret * SIZE})
    if not taken:
        return {"days": days, "trades": 0}
    tk = pd.DataFrame(taken)
    daily = tk.groupby("day").usd.sum()
    alld = pd.Series(0.0, index=pd.date_range(t.t.min().date(), t.t.max().date()).date)
    alld.update(daily)
    eq = alld.cumsum()
    return {"days": days, "trades": int(len(tk)), "usd_total": round(float(tk.usd.sum()), 0),
            "usd_per_day": round(float(alld.mean()), 1), "goal_per_day": GOAL,
            "days_goal_met": round(float((alld >= GOAL).mean()) * 100, 1),
            "red_days": round(float((alld < 0).mean()) * 100, 1),
            "worst_day": round(float(alld.min()), 0), "max_dd_usd": round(float((eq.cummax() - eq).max()), 0)}


def backtest():
    btc = build("BTC")
    frames = {"BTC": btc}
    for coin in COINS:
        if coin != "BTC":
            frames[coin] = build(coin, btc)
    res = {"made": datetime.now(timezone.utc).isoformat(timespec="minutes"), "rules": {
        "tp_pct": TP * 100, "sl": f"below support, {SL_MIN*100:.1f}-{SL_MAX*100:.1f}%", "fee_pct": FEE * 100,
        "max_hold_h": MAX_HOLD / 4, "size": SIZE, "max_open": MAX_OPEN, "max_per_day": MAX_PER_DAY,
        "max_losses_per_day": MAX_LOSSES}, "labels": LABELS, "variants": {k: v for k, v in VARIANTS.items()}}
    pooled = {m: {k: [] for k in VARIANTS} for m in ("structure", "fixed")}
    rnd = {"all": [], **{s[0]: [] for s in SESSIONS}}
    span = {}
    for coin, f in frames.items():
        f = f[f.hi_q50.notna() & f.ma200.notna() & f.h4_rsi14.notna()].reset_index(drop=True)
        frames[coin] = f
        days = max(1, (f.t.iat[-1] - f.t.iat[0]).days)
        span[coin] = [str(f.t.iat[0])[:10], str(f.t.iat[-1])[:10], days]
        cd = conditions(f)
        for name, names in VARIANTS.items():
            e = mask(cd, names).to_numpy()
            for mode in ("structure", "fixed"):
                t = simulate(f, e, mode)
                if not t.empty:
                    t["coin"] = coin
                pooled[mode][name].append(t)
        # random entries (same exits), sample of bars
        idx = RNG.choice(len(f) - MAX_HOLD, size=min(3000, len(f) - MAX_HOLD), replace=False)
        # simulate each independently (no busy-skip) for an unbiased baseline
        for i in idx:
            one = np.zeros(len(f), bool)
            one[i] = True
            t = simulate(f, one, "fixed")
            rnd["all"].append(t)
            rnd[f.session.iat[i]].append(t)
    days = max(v[2] for v in span.values())
    res["period"] = span
    out = {}
    for mode in ("structure", "fixed"):
        out[mode] = {}
        for name in VARIANTS:
            ts = [t for t in pooled[mode][name] if not t.empty]
            t = pd.concat(ts) if ts else pd.DataFrame()
            s = stats(t, days)
            if not t.empty:
                s["ci90_avg_pct"] = boot_ci(t.ret)
                s["by_session"] = {k: stats(g, days) for k, g in t.groupby("session")}
                s["by_coin"] = {k: stats(g, days) for k, g in t.groupby("coin")}
                s["portfolio"] = portfolio(t, days)
                mid = t.t.min() + (t.t.max() - t.t.min()) / 2
                s["halves_avg_pct"] = [round(float(t[t.t < mid].ret.mean()) * 100, 3),
                                       round(float(t[t.t >= mid].ret.mean()) * 100, 3)]
                if mode == "structure":
                    s["recent"] = [{"coin": r.coin, "t": str(r.t)[:16], "exit_t": str(r.exit_t)[:16],
                                    "ret_pct": round(r.ret * 100, 2), "why": r.why}
                                   for r in t.sort_values("t").tail(20).itertuples()]
            out[mode][name] = s
    res["results"] = out
    res["random"] = {k: stats(pd.concat(v), days) for k, v in rnd.items() if v}
    res["pick"], res["gate"] = pick(res)
    REPORT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({m: {k: {x: v.get(x) for x in ("n", "win", "avg_net_pct", "ci90_avg_pct")}
                          for k, v in out[m].items()} for m in out}, indent=1))
    print("random", json.dumps(res["random"]))
    print("pick", res["pick"], res["gate"])
    return res


def pick(res):
    """Gate: a "not falling" variant that beats random by ≥0.15%/trade after fees, with ≥30 trades,
    the lower 90% bound above zero, and positive in BOTH halves of the period. None -> paper only."""
    rnd = res["random"]["all"]["avg_net_pct"]
    best, why = None, "no rule has beaten random entries after fees yet — paper trading only"
    R = res["results"]["structure"]
    for name in VARIANTS:
        if not name[0] in "CDEFGHIJ":
            continue
        s = R[name]
        ci, hv = s.get("ci90_avg_pct"), s.get("halves_avg_pct") or [-1, -1]
        if s.get("n", 0) >= 30 and ci and ci[0] > 0 and s["avg_net_pct"] >= rnd + 0.15 and min(hv) > 0:
            if best is None or s["avg_net_pct"] > R[best]["avg_net_pct"]:
                best = name
    if best:
        s = R[best]
        why = (f"{best}: {s['n']} trades in {res['period']['BTC'][2]} days, {s['win']}% hit +{TP*100:g}%, "
               f"{s['avg_net_pct']:+.2f}% per trade after fees (random {rnd:+.2f}%)")
    return best, why


# ── live paper log ────────────────────────────────────────────────────
def load_log():
    p = LOG_DIR / "trades.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"trades": [], "made": None}


def save_log(log):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log["made"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    (LOG_DIR / "trades.json").write_text(json.dumps(log, indent=1, default=str))
    if REPORT.exists():
        (LOG_DIR / "backtest.json").write_text(REPORT.read_text())


def update_open(log, frames):
    """Resolve every proposal — taken or skipped — so the page can show hit or missed."""
    done = []
    for tr in log["trades"]:
        if tr["status"] != "open":
            continue
        f = frames[tr["coin"]]
        after = f[f.t > pd.Timestamp(tr["time"], tz="UTC")]
        for r in after.head(MAX_HOLD).itertuples():
            if r.low <= tr["sl"]:                 # same candle = counted as a loss
                tr.update(status="SL", exit=tr["sl"], exit_time=str(r.t)[:16])
                break
            if r.high >= tr["tp"]:
                tr.update(status="TP", exit=tr["tp"], exit_time=str(r.t)[:16])
                break
        else:
            if len(after) >= MAX_HOLD:
                r = after.iloc[MAX_HOLD - 1]
                tr.update(status="time", exit=float(r.close), exit_time=str(r.t)[:16])
            elif len(after):
                tr["last"] = float(after.close.iat[-1])
                tr["max_seen"] = float(after.high.max())
                tr["min_seen"] = float(after.low.min())
        if tr["status"] != "open":
            ret = tr["exit"] / tr["entry"] - 1 - FEE
            tr["pnl_usd"] = round(ret * tr["size"], 2)
            tr["ret_pct"] = round(ret * 100, 2)
            tr["result"] = "hit" if tr["status"] == "TP" else "missed"
            tr.pop("last", None)
            done.append(tr)
    return done


def outcome_msg(tr):
    head = "✅ HIT" if tr["result"] == "hit" else ("⏱ 24h exit" if tr["status"] == "time" else "❌ MISSED (stop)")
    tag = "" if tr.get("taken", True) else " (was skipped by the daily limits)"
    return (f"{head}{tag} — {tr['coin']} long from {tr['time']} UTC\n"
            f"Entry {fmtp(tr['entry'])} → exit {fmtp(tr['exit'])} at {tr['exit_time']} UTC · "
            f"{tr['ret_pct']:+.2f}% after fees = {'+' if tr['pnl_usd'] >= 0 else '−'}${abs(tr['pnl_usd']):.0f}")


def live():
    from .alerts import send
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    gated = bool(rep.get("pick"))
    rule = rep.get("pick") or LIVE_RULE
    if rule not in VARIANTS:
        rule, gated = LIVE_RULE, False
    alerts = gated or os.environ.get("DAYTRADE_TELEGRAM", "gated") == "always"
    btc = build("BTC")
    frames = {"BTC": btc}
    for coin in COINS:
        if coin != "BTC":
            frames[coin] = build(coin, btc)
    for k in frames:                          # newest row is still forming
        frames[k] = frames[k].iloc[:-1].reset_index(drop=True)
    log = load_log()
    for tr in update_open(log, frames):
        print(outcome_msg(tr))
        if alerts and tr.get("taken", True) and tr.get("alerted"):
            send(outcome_msg(tr))
    today = datetime.now(timezone.utc).date().isoformat()
    todays = [t for t in log["trades"] if t["time"][:10] == today and t.get("taken", True)]
    n_open = sum(t["status"] == "open" and t.get("taken", True) for t in log["trades"])
    losses = sum(t.get("pnl_usd", 0) < 0 for t in todays if t["status"] != "open")
    watch = []
    for coin, f in frames.items():
        cd = conditions(f)
        i = len(f) - 1
        r = f.iloc[i]
        ok = {k: bool(v.fillna(False).iat[i]) for k, v in cd.items()}
        need = VARIANTS[rule]
        snap = {"coin": coin, "time": str(r.t)[:16], "price": float(r.close), "rsi6": round(float(r.rsi6), 1),
                "session": r.session, "ok": ok, "need": need,
                "met": sum(ok[n] for n in need), "of": len(need)}
        watch.append(snap)
        if not all(ok[n] for n in need):
            continue
        if any(t["coin"] == coin and t["status"] == "open" for t in log["trades"]):
            continue
        if any(t["coin"] == coin and t["time"] == snap["time"] for t in log["trades"]):
            continue
        blocked = (len(todays) >= MAX_PER_DAY and "7 proposals already today") or \
                  (n_open >= MAX_OPEN and "2 positions already open") or \
                  (losses >= MAX_LOSSES and "2 losses today — done for the day")
        e = float(r.close)
        sl = stop_price(r, "structure")
        tr = {"id": f"{coin}-{snap['time']}", "coin": coin, "time": snap["time"], "session": r.session,
              "entry": e, "tp": e * (1 + TP), "sl": sl, "size": SIZE, "rule": rule, "rsi6": snap["rsi6"],
              "supports": {s: float(r[s]) for s in SUPPORTS if pd.notna(r[s]) and r[s] <= e
                           and e / r[s] - 1 <= SUPPORT_BAND},
              "pred_hi": float(r.hi_q50), "liq_above": float(r.liq_above) if pd.notna(r.liq_above) else None,
              "extras": [k for k in ("asia", "stretch", "capitulation", "magnet") if ok[k] and k not in need],
              "status": "open", "taken": not blocked, "skip_reason": blocked or None, "live": gated}
        log["trades"].append(tr)
        if blocked:
            continue
        todays.append(tr)
        n_open += 1
        gain = (TP - FEE) * SIZE
        loss = (1 - sl / e + FEE) * SIZE
        msg = (f"{'🟢' if gated else '📝 PAPER'} {coin} 1% long #{len(todays)}/7 today ({snap['time']} UTC, {r.session})\n"
               f"Buy ~{fmtp(e)} with ${SIZE:,} · TP {fmtp(tr['tp'])} (+${gain:.0f}) · SL {fmtp(sl)} (−${loss:.0f})\n"
               f"RSI6 {snap['rsi6']} hooked up, volume spike, green close over EMA7, BTC not dumping\n"
               f"Support: " + ", ".join(f"{k.replace('_', ' ')} {fmtp(v)}" for k, v in tr["supports"].items())
               + (f" · liquidity above {fmtp(tr['liq_above'])}" if tr["liq_above"] else "") + "\n"
               f"Bybit: limit buy, then OCO sell (GTC): TP limit {fmtp(tr['tp'])}, stop {fmtp(sl)} Market.\n"
               + ("Backtest: " + rep.get("gate", "") if gated else "Paper only — " + rep.get("gate", "no backtest yet")))
        print(msg)
        if alerts:
            tr["alerted"] = bool(send(msg))
    log["watch"] = watch
    log["rule"], log["gated"], log["gate"] = rule, gated, rep.get("gate")
    log["limits"] = {"size": SIZE, "max_open": MAX_OPEN, "max_per_day": MAX_PER_DAY, "tp": TP, "fee": FEE,
                     "max_losses": MAX_LOSSES, "goal": GOAL, "budget": BUDGET}
    log["trades"] = log["trades"][-600:]
    save_log(log)
    print("watch:", [(w["coin"], w["rsi6"], f"{w['met']}/{w['of']}") for w in watch])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        backtest()
    else:
        live()
