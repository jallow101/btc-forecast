"""Liquidity reports for Telegram: weekly outlook, daily brief, mid-week updates.

Style: two liquidity windows like a heatmap trader reads them
  * 7-day map  = 1h engine (low time frame)
  * 1-month map = 4h engine (mid time frame)
plus trend (MA50 rule used in the backtest), daily MA7/25 + RSI, the weekly
model's 1-week range, the US economic calendar and crypto headlines.

Usage: python -m btcf.liq_report {auto|weekly|daily|check} OUT_DIR
  auto: Monday 07 UTC -> weekly, other days 07 UTC -> daily, else -> check
"""
import json
import mimetypes
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import liquidity as L
from .alerts import send
from .paths import DATA

COINS = ["BTC", "ETH", "SOL", "WLD"]
NAMES = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "WLD": "Worldcoin"}
LOGS = {"BTC": DATA / "forecast_log.csv", "ETH": DATA / "eth" / "forecast_log.csv",
        "SOL": DATA / "sol" / "forecast_log.csv"}
PAGE = "https://jallow101.github.io/btc-forecast/liquidity.html"
UA = {"User-Agent": "Mozilla/5.0 (btc-forecast bot)"}
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FEEDS = {"CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
         "Cointelegraph": "https://cointelegraph.com/rss",
         "Decrypt": "https://decrypt.co/feed"}
KEYS = {  # headline keyword -> weight
    "etf": 3, "fed": 3, "fomc": 3, "powell": 3, "rate": 2, "inflation": 2, "cpi": 3, "jobs": 2, "payroll": 3,
    "sec": 2, "hack": 4, "exploit": 4, "binance": 2, "liquidat": 2, "outflow": 2, "inflow": 2, "tether": 2,
    "stablecoin": 1, "ban": 2, "regulat": 1, "lawsuit": 2, "strategy": 1, "saylor": 1, "treasury": 1,
    "tariff": 2, "ipo": 1, "anthropic": 1, "nasdaq": 1, "unlock": 2, "whale": 1, "miners": 1,
    "worldcoin": 3, "world network": 2, "wld": 3, "solana": 2, "ethereum": 2, "ether ": 2, "bitcoin": 1,
}
SEVERE = ["hack", "exploit", "insolven", "halts withdrawals", "suspends withdrawals", "emergency", "depeg",
          "sues", "delist", "bankrupt", "ban on", "bans "]


def fmt(p):
    return f"${p:,.0f}" if p >= 1000 else f"${p:,.2f}" if p >= 10 else f"${p:.4f}"


def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f}%"


# ── data ───────────────────────────────────────────────────────────────
def snap(out, coin, tf):
    p = out / f"{coin}_{tf}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ta(coin):
    """Daily MA7/MA25 and RSI6 from the 4h Hyperliquid candles."""
    d = L.load(coin, "4h").set_index("t")
    day = d.close.resample("1D").last().dropna()
    def rsi(s, n):
        x = s.diff()
        g = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        lo = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + g / lo)
    return {"d_ma7": float(day.rolling(7).mean().iat[-1]), "d_ma25": float(day.rolling(25).mean().iat[-1]),
            "d_rsi6": float(rsi(day, 6).iat[-1]), "chg24": float(d.close.iat[-1] / d.close.iat[-7] - 1)}


def model_band(coin):
    p = LOGS.get(coin)
    if not p or not p.exists():
        return None
    f = pd.read_csv(p)
    r = f[(f.run_date == f.run_date.max()) & (f.h == 1)]
    if r.empty:
        return None
    r = r.iloc[0]
    return {"p10": r.p10, "median": r["median"], "p90": r.p90, "week": r.week_ending}


def _get(url, timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def calendar(days_ahead=7):
    """High-impact USD events from the Forex Factory weekly feed."""
    try:
        ev = json.loads(_get(CAL_URL))
    except Exception as e:  # noqa: BLE001
        print("calendar unavailable:", e)
        return None
    now = datetime.now(timezone.utc)
    out = []
    for e in ev:
        if e.get("country") != "USD" or e.get("impact") != "High":
            continue
        try:
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        except Exception:  # noqa: BLE001
            continue
        if now - timedelta(hours=2) <= t <= now + timedelta(days=days_ahead):
            out.append({"t": t, "title": e.get("title", ""), "forecast": e.get("forecast", ""),
                        "previous": e.get("previous", "")})
    return sorted(out, key=lambda x: x["t"])


def headlines(hours=48, n=6):
    items = []
    for src, url in FEEDS.items():
        try:
            root = ET.fromstring(_get(url))
        except Exception as e:  # noqa: BLE001
            print("feed unavailable:", src, e)
            continue
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            try:
                t = parsedate_to_datetime(it.findtext("pubDate")).astimezone(timezone.utc)
            except Exception:  # noqa: BLE001
                continue
            if datetime.now(timezone.utc) - t > timedelta(hours=hours):
                continue
            low = " " + title.lower() + " "
            score = sum(w for k, w in KEYS.items() if k in low)
            severe = any(k in low for k in SEVERE)
            if score >= 3 or severe:
                items.append({"src": src, "title": title, "link": link, "t": t, "score": score + 5 * severe,
                              "severe": severe})
    # drop near-duplicate titles
    seen, out = set(), []
    for x in sorted(items, key=lambda x: (-x["score"], -x["t"].timestamp())):
        key = re.sub(r"[^a-z]", "", x["title"].lower())[:40]
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out[:n]


# ── analysis ───────────────────────────────────────────────────────────
def heavy_side(s, ratio=1.5):
    a = (s["majors"].get("above") or [None])[0]
    b = (s["majors"].get("below") or [None])[0]
    if not a or not b:
        return ("above" if a else "below" if b else None), a, b
    if a["weight"] > ratio * b["weight"]:
        return "above", a, b
    if b["weight"] > ratio * a["weight"]:
        return "below", a, b
    return None, a, b


def plan(coin, out):
    s1, s4 = snap(out, coin, "1h"), snap(out, coin, "4h")
    if not s1 or not s4:
        return None
    price, trend = s1["price"], s4["trend"]
    h4, a4, b4 = heavy_side(s4)
    h1, a1, b1 = heavy_side(s1)
    heavy = h4 or h1
    t = ta(coin)
    tdir = {"up": "above", "down": "below"}.get(trend)
    if tdir and heavy == tdir:
        case = "with"
    elif tdir and heavy and heavy != tdir:
        case = "against"
    else:
        case = "range"
    up1 = a1["price"] if a1 else None
    dn1 = b1["price"] if b1 else None
    up4 = a4["price"] if a4 else None
    dn4 = b4["price"] if b4 else None
    lines = []
    if case == "with" and trend == "down":
        lines.append(f"A (main, backtest edge): trend and liquidity both point down → sweep toward {fmt(dn1 or dn4)}"
                     + (f", then {fmt(dn4)}" if dn4 and dn1 and dn4 < dn1 * 0.995 else ""))
        lines.append(f"B: relief bounce into {fmt(up1)} first, then lower" if up1 else "B: chop near current level, then lower")
        inval = up4 or up1
        inval_txt = f"4h close above {fmt(inval)} cancels the bearish plan" if inval else "—"
    elif case == "with" and trend == "up":
        lines.append(f"A (main, backtest edge): trend and liquidity both point up → run toward {fmt(up1 or up4)}"
                     + (f", then {fmt(up4)}" if up4 and up1 and up4 > up1 * 1.005 else ""))
        lines.append(f"B: dip into {fmt(dn1)} first to clear longs, then higher" if dn1 else "B: chop, then higher")
        inval = dn4 or dn1
        inval_txt = f"4h close below {fmt(inval)} cancels the bullish plan" if inval else "—"
    elif case == "against":
        mag = (up4 or up1) if heavy == "above" else (dn4 or dn1)
        trend_t = dn1 if trend == "down" else up1
        lines.append(f"A (trend): {trend}trend continues toward {fmt(trend_t)}" if trend_t else f"A (trend): {trend}trend continues")
        lines.append(f"B (magnet, weaker): liquidity is heavier {heavy} → squeeze toward {fmt(mag)} if price "
                     f"{'reclaims the daily MA7 ' + fmt(t['d_ma7']) if heavy == 'above' else 'loses the daily MA7 ' + fmt(t['d_ma7'])}")
        inval = t["d_ma7"]
        inval_txt = (f"Daily close {'above' if heavy == 'above' else 'below'} {fmt(inval)} switches to plan B. "
                     "Trend and liquidity disagree: smaller size")
    elif tdir:
        tgt = dn1 if trend == "down" else up1
        other = up1 if trend == "down" else dn1
        lines.append(f"A (trend): liquidity is balanced, so the {trend}trend leads → toward {fmt(tgt)}" if tgt else f"A (trend): {trend}trend continues")
        if other:
            lines.append(f"B: a counter-move into {fmt(other)} first is possible; it is the edge of the 7-day range")
        inval = other
        inval_txt = f"4h close {'above' if trend == 'down' else 'below'} {fmt(other)} breaks the {trend}trend structure" if other else "—"
    else:
        lines.append(f"Range: {fmt(dn1)} – {fmt(up1)} (7-day clusters). Nearest one usually gets hit first" if up1 and dn1 else "No clear setup")
        lines.append("No 4h trend: fade the range edges, or wait for a trend before trusting a direction")
        inval = None
        inval_txt = "Break and 4h close outside the range sets the next direction"
    adds = [x for x in (dn1, dn4) if x]
    return {"coin": coin, "price": price, "trend": trend, "heavy": heavy, "case": case, "ta": t,
            "ltf": {"up": a1, "dn": b1}, "mtf": {"up": a4, "dn": b4},
            "lines": lines, "inval": inval, "inval_txt": inval_txt, "adds": sorted(set(adds), reverse=True),
            "band": model_band(coin)}


def coin_block(p, full=True):
    t = p["ta"]
    r = []
    rsi = t["d_rsi6"]
    rtxt = " (very oversold)" if rsi < 15 else " (oversold)" if rsi < 30 else " (overbought)" if rsi > 70 else ""
    r.append(f"{p['coin']} {fmt(p['price'])} ({t['chg24'] * 100:+.1f}% 24h) · 4h {p['trend']}trend · "
             f"daily RSI6 {rsi:.0f}{rtxt}")
    r.append(f"Daily MA7 {fmt(t['d_ma7'])} · MA25 {fmt(t['d_ma25'])}")
    def cl(c):
        return f"{fmt(c['price'])} ({pct(c['price'], p['price'])}, {c['weight']:.1f})" if c else "none"
    r.append(f"7-day map: above {cl(p['ltf']['up'])} | below {cl(p['ltf']['dn'])}")
    if full:
        r.append(f"1-month map: above {cl(p['mtf']['up'])} | below {cl(p['mtf']['dn'])}")
    verdict = {"with": "agrees with the trend (the case with a backtest edge)",
               "against": "is against the trend (weak edge)", "range": "gives no clear lean"}[p["case"]]
    r.append(f"→ Liquidity heavier {p['heavy'] or 'on neither side'}; it {verdict}")
    if full:
        r += ["• " + x for x in p["lines"]]
        r.append("• Invalidation: " + p["inval_txt"])
        if p["adds"]:
            r.append("• Buy-the-dip zones (liquidity below): " + ", ".join(fmt(x) for x in p["adds"]))
        if p["band"]:
            b = p["band"]
            r.append(f"• Model, week to {b['week']}: {fmt(b['p10'])}–{fmt(b['p90'])} (median {fmt(b['median'])})")
    return "\n".join(r)


def cal_block(cal, days=7):
    if cal is None:
        return "Calendar: unavailable this run"
    if not cal:
        return "Calendar: no high-impact US data in this window"
    rows = [f"• {e['t']:%a %H:%M} {e['title']}" + (f" (fcst {e['forecast']})" if e['forecast'] else "")
            for e in cal if e["t"] <= datetime.now(timezone.utc) + timedelta(days=days)]
    return "US data to watch (UTC, high impact):\n" + ("\n".join(rows[:10]) if rows else "• none")


def news_block(news):
    if not news:
        return "Headlines: nothing major flagged"
    return "Headlines that could move it:\n" + "\n".join(
        f"• {'⚠️ ' if x['severe'] else ''}{x['title']} ({x['src']})" for x in news)


# ── chart ──────────────────────────────────────────────────────────────
def chart(out, coin, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=110, facecolor="#141414")
    for ax, tf, name in ((axes[0], "1h", "7-day map (1h)"), (axes[1], "4h", "1-month map (4h)")):
        s = snap(out, coin, tf)
        ax.set_facecolor("#141414")
        if not s:
            continue
        C = np.array([c[1:] for c in s["candles"]], float)
        n = len(C)
        W = np.array(s["heat"]["w"], float).T
        lo, step, nb = s["grid"]["lo"], s["grid"]["step"], s["grid"]["n"]
        vmax = np.percentile(W[W > 0], 99.5) if (W > 0).any() else 1
        ax.imshow(np.sqrt(np.clip(W / vmax, 0, 1)), aspect="auto", origin="lower", cmap="inferno",
                  extent=[0, n, lo, lo + nb * step], interpolation="nearest")
        for i, (o, h, l, c) in enumerate(C):
            col = "#5fd38d" if c >= o else "#ff6b6b"
            ax.plot([i + .5, i + .5], [l, h], color=col, lw=.6)
            ax.add_patch(plt.Rectangle((i + .2, min(o, c)), .6, max(abs(c - o), step * .05), color=col))
        price = s["price"]
        pmin = max(lo, min(C[:, 2].min(), price * (1 - 0.12)))
        pmax = min(lo + nb * step, max(C[:, 1].max(), price * 1.12))
        ax.set_ylim(pmin, pmax)
        ax.set_xlim(0, n + 1)
        ax.axhline(price, color="white", lw=.8, ls="--")
        for side, col in (("above", "#ffd166"), ("below", "#8ecae6")):
            for k, c in enumerate(s["majors"].get(side, [])[:2]):
                if pmin < c["price"] < pmax:
                    ax.axhline(c["price"], color=col, lw=.8, alpha=.8 if k == 0 else .45)
                    ax.text(n + .5, c["price"], f" {fmt(c['price'])}", color=col, va="center", fontsize=8,
                            ha="right", bbox=dict(fc="#141414", ec="none", alpha=.7))
        ax.set_title(f"{coin} {name} · {fmt(price)} · {s['trend']}trend", color="white", fontsize=11)
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        ticks = [0, n // 2, n - 1]
        ax.set_xticks([t + .5 for t in ticks], [s["candles"][t][0][5:10] for t in ticks])
        ax.yaxis.tick_right()
        for sp in ax.spines.values():
            sp.set_color("#333")
    fig.text(.01, .01, "Brighter = more estimated liquidity (liquidations at 10-100x + swing stops). "
             "Yellow = biggest above, blue = biggest below. Estimate from candles, not exchange data.",
             color="#999", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def send_photo(path, caption=""):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(f"[photo not sent: no Telegram secrets] {path}")
        return False
    chat = chat.strip().strip('"').strip("'")
    b = uuid.uuid4().hex
    parts = []
    for k, v in (("chat_id", chat), ("caption", caption[:1000])):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    data = Path(path).read_bytes()
    ctype = mimetypes.guess_type(str(path))[0] or "image/png"
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="photo"; filename="{Path(path).name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(f"https://api.telegram.org/bot{token.strip()}/sendPhoto", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status == 200
    except Exception as e:  # noqa: BLE001
        print("photo send failed:", e)
        return False


def send_long(text):
    """Telegram caps a message at 4096 characters."""
    chunks, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 > 3800:
            chunks.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + para
    if cur:
        chunks.append(cur)
    for c in chunks:
        send(c)
        time.sleep(1)


# ── reports ────────────────────────────────────────────────────────────
DISCLAIMER = (f"Liquidity = estimates from candles (backtest: useful as targets when they agree with the trend; "
              f"buying a sweep alone did not work). Not financial advice.\n{PAGE}")


def weekly(out, title="📊 Weekly liquidity outlook", why=None):
    plans = {c: plan(c, out) for c in COINS}
    cal, news = calendar(7), headlines(72)
    now = datetime.now(timezone.utc)
    head = f"{title} — week of {now:%d %b %Y}"
    if why:
        head += "\nWhat changed: " + "; ".join(why)
    body = [head]
    for c in COINS:
        if plans[c]:
            body.append(coin_block(plans[c], full=True))
    body += [cal_block(cal), news_block(news), DISCLAIMER]
    text = "\n\n".join(body)
    print(text)
    img = chart(out, "BTC", out / "btc_weekly.png")
    send_photo(img, f"{title}: BTC 7-day and 1-month liquidity maps")
    send_long(text)
    (out / "weekly_plan.json").write_text(json.dumps({
        "made": now.isoformat(), "revised": bool(why),
        "plans": {c: {k: p[k] for k in ("price", "trend", "heavy", "case", "inval")} | {
            "mtf_up": (p["mtf"]["up"] or {}).get("price"), "mtf_dn": (p["mtf"]["dn"] or {}).get("price"),
            "mtf_up_w": (p["mtf"]["up"] or {}).get("weight"), "mtf_dn_w": (p["mtf"]["dn"] or {}).get("weight")}
            for c, p in plans.items() if p},
        "text": text}, indent=1, default=str))
    return text


def daily(out):
    plans = {c: plan(c, out) for c in COINS}
    cal, news = calendar(1), headlines(24, 4)
    now = datetime.now(timezone.utc)
    prevp = out / "weekly_plan.json"
    wk = json.loads(prevp.read_text()) if prevp.exists() else None
    body = [f"☀️ Daily liquidity brief — {now:%a %d %b}"]
    for c in COINS:
        p = plans[c]
        if not p:
            continue
        txt = coin_block(p, full=False)
        if wk and c in wk["plans"]:
            w = wk["plans"][c]
            txt += f"\n• vs weekly plan: price {pct(p['price'], w['price'])}, trend {w['trend']}→{p['trend']}"
        body.append(txt)
    body += [cal_block(cal, 1), news_block(news), PAGE]
    text = "\n\n".join(body)
    print(text)
    img = chart(out, "BTC", out / "btc_daily.png")
    send_photo(img, "Daily: BTC liquidity maps")
    send_long(text)
    return text


def check(out):
    """Hourly: big shift since the weekly plan -> send a revised plan (max once per 24h)."""
    prevp = out / "weekly_plan.json"
    if not prevp.exists():
        print("no weekly plan yet")
        return None
    wk = json.loads(prevp.read_text())
    st_p = out / "report_state.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    if time.time() - st.get("last_update", 0) < 24 * 3600 or \
            datetime.now(timezone.utc) - datetime.fromisoformat(wk["made"]) < timedelta(hours=12):
        print("shift check: cooling down")
        return None
    why = []
    for c in ("BTC", "ETH"):
        w = wk["plans"].get(c)
        s4 = snap(out, c, "4h")
        if not w or not s4:
            continue
        price = s4["price"]
        h, a, b = heavy_side(s4)
        if s4["trend"] != w["trend"] and s4["trend"] != "none":
            why.append(f"{c} 4h trend turned {s4['trend']}")
        if h and w["heavy"] and h != w["heavy"]:
            why.append(f"{c} liquidity flipped: now heavier {h}")
        if w["inval"] and ((w["case"] == "with" and w["trend"] == "down" and price > w["inval"]) or
                           (w["case"] == "with" and w["trend"] == "up" and price < w["inval"])):
            why.append(f"{c} broke the invalidation {fmt(w['inval'])}")
        if abs(price / w["price"] - 1) > 0.06:
            why.append(f"{c} moved {pct(price, w['price'])} since the plan")
        for side, key in (("above", "mtf_up"), ("below", "mtf_dn")):
            cur = (s4["majors"].get(side) or [None])[0]
            if cur and w.get(key) and abs(cur["price"] / w[key] - 1) > 0.03 and cur["weight"] > 1.5 * (w.get(key + "_w") or 0):
                why.append(f"{c} new big cluster {side} at {fmt(cur['price'])}")
    severe = [x for x in headlines(6, 10) if x["severe"] and x["link"] not in st.get("news_sent", [])]
    for x in severe[:2]:
        why.append(f"news: {x['title']}")
    if not why:
        print("shift check: nothing big")
        return None
    st["last_update"] = time.time()
    st["news_sent"] = (st.get("news_sent", []) + [x["link"] for x in severe])[-50:]
    st_p.write_text(json.dumps(st))
    return weekly(out, title="📣 Mid-week update", why=why)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/data/liquidity")
    if mode == "auto":
        now = datetime.now(timezone.utc)
        mode = "weekly" if now.weekday() == 0 and now.hour == 7 else "daily" if now.hour == 7 else "check"
    {"weekly": weekly, "daily": daily, "check": check}[mode](out)


if __name__ == "__main__":
    main()
