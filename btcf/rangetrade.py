"""Range trades: buy the model's predicted daily low ONLY when the setup is right.

Rule (tested on ~400 days of 15m data, BTC/ETH/SOL; see backtest()):
  Setup  - oversold: daily RSI(14) < 35 at yesterday's close, or 4h RSI(14) < 30
         - liquidity magnet up: more estimated liquidation/stop liquidity in the
           6% above price than in the 6% below (1h liquidity engine)
  Entry  - limit buy at today's predicted low (median), if the setup holds when it fills
  Exit   - take profit at the bottom of today's predicted-high range (hi_q10)
         - stop 1% under the bottom of the predicted-low range (lo_q10)
         - "not moving" exit: 12h after the fill, if price isn't 0.4% above entry, close
         - otherwise hold up to 3 days
Every day without the setup is a day off.

live(): arms a morning alert when the setup is on, paper-logs fills and exits
to out/trades/range_log.json (published with the trade log) and tells Telegram.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import liquidity as L
from .daytrade import load_15m, resample
from .paths import DATA, ROOT
from .scalp import fmtp, rsi

COINS = {"BTC": "btc", "ETH": "eth", "SOL": "sol"}
FEE = 0.002
SIZE = 2500
WEAK_BARS, WEAK_MIN = 48, 0.004
MAX_BARS = 3 * 96
Q = ("hi_q10", "hi_q50", "lo_q10", "lo_q50")
LOG = ROOT / "out" / "trades" / "range_log.json"
REPORT = DATA / "range_backtest.json"


# ── data ──────────────────────────────────────────────────────────────
def preds(key):
    out = {}
    p = DATA / f"daily_bt_{key}.csv"
    if p.exists():
        b = pd.read_csv(p, parse_dates=["target_date"])
        for r in b[b.k == 1].itertuples():
            out[r.target_date.date()] = {q: r.close * np.exp(getattr(r, q) * r.scale) for q in Q}
    p = DATA / "daily_log.csv"
    if p.exists():
        lg = pd.read_csv(p)
        lg = lg[(lg.asset == key) & ((pd.to_datetime(lg.target_date) - pd.to_datetime(lg.run_date)).dt.days == 1)]
        for r in lg.itertuples():
            out[pd.Timestamp(r.target_date).date()] = {q: getattr(r, q) for q in Q}
    p = DATA / "daily_range_latest.json"
    if p.exists():
        for r in json.loads(p.read_text()).get(key, []):
            d = pd.Timestamp(r["target_date"]).date()
            out.setdefault(d, {q: r[q] for q in Q})
    return out


def liq_ctx(h1):
    up = np.full(len(h1), np.nan)
    dn = np.full(len(h1), np.nan)
    cl = h1.close.to_numpy()

    def cb(i, eng, swept):
        px, w, _, _ = eng.weights(i)
        if len(px):
            p = cl[i]
            up[i] = w[(px > p * 1.002) & (px < p * 1.06)].sum()
            dn[i] = w[(px < p * 0.998) & (px > p * 0.94)].sum()

    L.run_engine(h1.reset_index(drop=True), cb)
    return pd.DataFrame({"liq_up": up, "liq_dn": dn}, index=h1.t + pd.Timedelta("1h"))


def features(coin):
    f = load_15m(coin).reset_index(drop=True)
    h1, h4, d1 = resample(f, "1h"), resample(f, "4h"), resample(f, "1D")
    f["ct"] = f.t + pd.Timedelta("15min")
    c4 = pd.DataFrame({"rsi4h": rsi(h4.close, 14).to_numpy()}, index=h4.t + pd.Timedelta("4h"))
    for c in (liq_ctx(h1), c4):
        f = pd.merge_asof(f, c.sort_index(), left_on="ct", right_index=True)
    dd = pd.DataFrame({"rsi_d": rsi(d1.close, 14).to_numpy()}, index=(d1.t + pd.Timedelta("1D")).dt.date)
    f["d"] = f.t.dt.date
    f = f.join(dd, on="d")
    f["oversold"] = (f.rsi_d < 35) | (f.rsi4h < 30)
    f["magnet_up"] = (f.liq_up > f.liq_dn).fillna(False)
    f["setup"] = f.oversold & f.magnet_up
    return f


def manage(f, i, entry, p, last=None):
    """Walk bars after the fill: returns (result, exit_price, exit_index) or (None, None, last_index)."""
    H, Lo, C = f.high.to_numpy(), f.low.to_numpy(), f.close.to_numpy()
    tp, stop = p["hi_q10"], p["lo_q10"] * 0.99
    end = min(i + MAX_BARS, len(C) - 1 if last is None else last)
    for j in range(i + 1, end + 1):
        if Lo[j] <= stop:
            return "stop", stop, j
        if H[j] >= tp:
            return "TP", tp, j
        if j - i == WEAK_BARS and C[j] < entry * (1 + WEAK_MIN):
            return "not moving", C[j], j
    if i + MAX_BARS <= end:
        return "3 days", C[end], end
    return None, None, end


# ── backtest ──────────────────────────────────────────────────────────
def backtest():
    rows = []
    span = {}
    for coin, key in COINS.items():
        dp, f = preds(key), features(coin)
        Lo = f.low.to_numpy()
        span[coin] = [str(f.d.iat[0]), str(f.d.iat[-1])]
        busy = -1
        for d, g in f.groupby("d"):
            p = dp.get(d)
            a, b = g.index[0], g.index[-1]
            if not p or b - a < 90 or a <= busy:
                continue
            hit = np.flatnonzero(Lo[a:b + 1] <= p["lo_q50"])
            if not len(hit):
                continue
            i = a + hit[0]
            base = {"coin": coin, "d": str(d), "setup": bool(f.setup.iat[i]), "oversold": bool(f.oversold.iat[i]),
                    "magnet_up": bool(f.magnet_up.iat[i])}
            why, px, k = manage(f, i, p["lo_q50"], p)
            if why is None:
                continue
            rows.append({**base, "why": why, "ret": px / p["lo_q50"] - 1 - FEE, "exit_d": str(f.d.iat[k])})
            if base["setup"]:
                busy = k
    R = pd.DataFrame(rows)

    def st(s):
        if s.empty:
            return {"n": 0}
        return {"n": int(len(s)), "avg_pct": round(float(s.ret.mean()) * 100, 2),
                "win_pct": round(float((s.ret > 0).mean()) * 100), "tp_pct": round(float((s.why == "TP").mean()) * 100),
                "usd_total_2500": round(float(s.ret.sum()) * SIZE), "by_exit": s.why.value_counts().to_dict(),
                "by_coin": {c: round(float(v) * 100, 2) for c, v in s.groupby("coin").ret.mean().items()}}

    S = R[R.setup]
    rng = np.random.default_rng(1)
    ci = (np.percentile([rng.choice(S.ret, len(S)).mean() for _ in range(3000)], [5, 95]) * 100).round(2).tolist() \
        if len(S) >= 10 else None
    rep = {"made": datetime.now(timezone.utc).isoformat(timespec="minutes"), "period": span,
           "every_fill": st(R), "setup_only": {**st(S), "ci90_avg_pct": ci},
           "oversold_only": st(R[R.oversold]), "magnet_only": st(R[R.magnet_up]),
           "setup_trades": S.tail(30).assign(ret=lambda x: (x.ret * 100).round(2)).to_dict("records")}
    REPORT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "setup_trades"}, indent=1, default=str))
    return rep


# ── live ──────────────────────────────────────────────────────────────
def live():
    from .alerts import send
    log = json.loads(LOG.read_text()) if LOG.exists() else {"trades": [], "armed": {}}
    today = datetime.now(timezone.utc).date()
    status = {}
    for coin, key in COINS.items():
        p = preds(key).get(today)
        f = features(coin).iloc[:-1].reset_index(drop=True)          # closed candles only
        last = f.iloc[-1]
        # manage open paper positions
        for tr in log["trades"]:
            if tr["coin"] != coin or tr["status"] != "open":
                continue
            ft = pd.Timestamp(tr["fill_time"])
            ft = ft.tz_localize("UTC") if ft.tzinfo is None else ft
            m = f.index[f.t == ft]
            if not len(m):
                continue
            i = int(m[0])
            why, px, k = manage(f, i, tr["entry"], tr["plan"], last=len(f) - 1)
            tr["last"] = float(last.close)
            if why:
                ret = px / tr["entry"] - 1 - FEE
                tr.update(status=why, exit=float(px), exit_time=str(f.t.iat[k])[:16], ret_pct=round(ret * 100, 2),
                          pnl_usd=round(ret * SIZE, 2), result="hit" if why == "TP" else "missed")
                send(f"{'✅' if why == 'TP' else '❌' if why == 'stop' else '⏹'} RANGE {coin} closed ({why}) — "
                     f"entry {fmtp(tr['entry'])} → {fmtp(px)} = {ret*100:+.2f}% (${ret*SIZE:+.0f} on ${SIZE:,}, paper)")
        if not p:
            continue
        setup = bool(last.setup)
        status[coin] = {"price": float(last.close), "rsi_d": round(float(last.rsi_d), 1), "rsi4h": round(float(last.rsi4h), 1),
                        "liq_up": round(float(last.liq_up or 0), 2), "liq_dn": round(float(last.liq_dn or 0), 2),
                        "oversold": bool(last.oversold), "magnet_up": bool(last.magnet_up), "setup": setup,
                        "buy": p["lo_q50"], "tp": p["hi_q10"], "stop": p["lo_q10"] * 0.99, "time": str(last.t)[:16]}
        akey = f"{coin}-{today}"
        open_ = any(t["coin"] == coin and t["status"] == "open" for t in log["trades"])
        if setup and not open_ and akey not in log["armed"] and last.close > p["lo_q50"]:
            log["armed"][akey] = str(last.t)[:16]
            send(f"🎯 RANGE SETUP ON — {coin} ({today})\n"
                 f"Oversold (daily RSI {last.rsi_d:.0f}, 4h RSI {last.rsi4h:.0f}) and more liquidity above than below.\n"
                 f"Limit buy {fmtp(p['lo_q50'])} (today's predicted low) · TP {fmtp(p['hi_q10'])} · stop {fmtp(p['lo_q10']*0.99)}\n"
                 f"If filled: close it if it isn't up 0.4% after 12h; hold max 3 days.\n"
                 f"Unproven rule (28 trades in the backtest) — paper-tracked on the trades page.")
        # paper fill: today's first closed candle that reached the predicted low, with the setup on
        if not open_:
            g = f[(f.d == today) & (f.low <= p["lo_q50"])]
            done = any(t["coin"] == coin and t["day"] == str(today) for t in log["trades"])
            if len(g) and not done and bool(g.setup.iat[0]):
                r = g.iloc[0]
                tr = {"coin": coin, "day": str(today), "fill_time": str(r.t), "entry": float(p["lo_q50"]),
                      "plan": {q: float(p[q]) for q in Q}, "tp": float(p["hi_q10"]), "stop": float(p["lo_q10"] * 0.99),
                      "status": "open", "size": SIZE}
                log["trades"].append(tr)
                send(f"🟢 RANGE {coin} filled (paper) at {fmtp(tr['entry'])} ({str(r.t)[11:16]} UTC) · "
                     f"TP {fmtp(tr['tp'])} · stop {fmtp(tr['stop'])} · exit if not +0.4% by 12h")
    log["status"] = status
    log["made"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    log["armed"] = {k: v for k, v in log["armed"].items() if k[-10:] >= str(today - pd.Timedelta(days=7))}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, indent=1, default=str))
    if REPORT.exists():
        (LOG.parent / "range_backtest.json").write_text(REPORT.read_text())
    print({c: (s["setup"], s["oversold"], s["magnet_up"]) for c, s in status.items()})


if __name__ == "__main__":
    import sys
    backtest() if len(sys.argv) > 1 and sys.argv[1] == "backtest" else live()
