"""Real leverage data from OKX (free, no key): funding, open interest,
long/short account ratio and actual liquidations.

The liquidity heatmap estimates where liquidations sit from candles alone.
This adds what traders actually hold and where they actually got wiped.
Liquidation prints are kept for 7 days (OKX only serves recent ones).
Writes out/trades/derivs.json (published with the trade log).
"""
import json
import time
import urllib.request
from datetime import datetime, timezone

OUT = None  # set in update()
COINS = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP", "SOL": "SOL-USDT-SWAP"}
CTVAL = {"BTC": 0.01, "ETH": 0.1, "SOL": 1.0}
BASE = "https://www.okx.com"
KEEP = 7 * 86400


def _get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "btc-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        js = json.load(r)
    if js.get("code") not in ("0", 0):
        raise RuntimeError(f"{path}: {js.get('msg')}")
    return js["data"]


def liquidations(coin, inst, old):
    """New liquidation prints merged into the stored list [ts, price, usd, side]."""
    seen = {(r[0], r[1], r[2]) for r in old}
    out = list(old)
    after = ""
    for _ in range(5):                           # up to 500 newest prints per run
        data = _get(f"/api/v5/public/liquidation-orders?instType=SWAP&instFamily={inst[:-5]}"
                    f"&state=filled&limit=100" + (f"&after={after}" if after else ""))
        det = [d for blk in data for d in blk.get("details", [])]
        if not det:
            break
        new = 0
        for d in det:
            ts = int(d["ts"]) // 1000
            px = float(d["bkPx"])
            usd = round(float(d["sz"]) * CTVAL[coin] * px, 0)
            side = d.get("posSide") or ("long" if d.get("side") == "sell" else "short")
            key = (ts, px, usd)
            if key not in seen:
                seen.add(key)
                out.append([ts, px, usd, side])
                new += 1
        after = det[-1]["ts"]
        if new < len(det) * 0.5:                 # reached what we already had
            break
        time.sleep(0.2)
    cut = time.time() - KEEP
    return sorted(r for r in out if r[0] >= cut)


def coin_block(coin, inst, old_liqs):
    fr = _get(f"/api/v5/public/funding-rate?instId={inst}")[0]
    funding = float(fr["fundingRate"])
    oi = _get(f"/api/v5/rubik/stat/contracts/open-interest-volume?ccy={coin}&period=1H")
    oi = sorted([int(r[0]) // 1000, float(r[1])] for r in oi)       # [ts, OI usd]
    ls = _get(f"/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={coin}&period=1H")
    ls = sorted([int(r[0]) // 1000, float(r[1])] for r in ls)
    px = float(_get(f"/api/v5/market/ticker?instId={inst}")[0]["last"])
    liqs = liquidations(coin, inst, old_liqs)
    now = time.time()
    day = [r for r in liqs if r[0] >= now - 86400]
    long_usd = sum(r[2] for r in day if r[3] == "long")
    short_usd = sum(r[2] for r in day if r[3] == "short")
    # where did the liquidations cluster (0.25% bins), last 24h
    step = px * 0.0025
    bins = {}
    for r in day:
        b = round(r[1] / step)
        k = (b, r[3])
        bins[k] = bins.get(k, 0) + r[2]
    clusters = sorted(([round(b * step, 2), usd, s] for (b, s), usd in bins.items()), key=lambda x: -x[1])[:6]

    def chg(series, hours):
        if len(series) < hours + 1:
            return None
        a, b = series[-hours - 1][1], series[-1][1]
        return (b / a - 1) * 100 if a else None

    blk = {"price": px, "funding_8h_pct": round(funding * 100, 4),
           "funding_year_pct": round(funding * 3 * 365 * 100, 1),
           "oi_usd": oi[-1][1] if oi else None, "oi_chg_24h_pct": chg(oi, 24), "oi_chg_4h_pct": chg(oi, 4),
           "ls_ratio": ls[-1][1] if ls else None, "ls_ratio_24h_ago": ls[-25][1] if len(ls) > 24 else None,
           "liq_24h": {"long_usd": long_usd, "short_usd": short_usd, "n": len(day)},
           "liq_clusters_24h": clusters, "oi_series": oi[-72:], "ls_series": ls[-72:]}
    blk["read"] = read(blk)
    return blk, liqs


def read(b):
    """Plain-language reading of the numbers."""
    out = []
    f = b["funding_8h_pct"]
    if f >= 0.03:
        out.append(f"Funding high (+{f:.3f}%/8h): longs are crowded and paying — a flush down is more likely than a squeeze up.")
    elif f <= -0.005:
        out.append(f"Funding negative ({f:.3f}%/8h): shorts are crowded — fuel for a squeeze up.")
    else:
        out.append(f"Funding normal ({f:+.3f}%/8h): no side is badly crowded.")
    oc = b.get("oi_chg_24h_pct")
    if oc is not None:
        if oc >= 3:
            out.append(f"Open interest +{oc:.1f}% in 24h: new leverage piling in — bigger moves and liquidation cascades get more likely.")
        elif oc <= -3:
            out.append(f"Open interest {oc:.1f}% in 24h: leverage is being flushed out — moves after a flush are often cleaner.")
        else:
            out.append(f"Open interest {oc:+.1f}% in 24h: leverage steady.")
    r = b.get("ls_ratio")
    if r:
        if r >= 2:
            out.append(f"Long/short accounts {r:.2f}: most traders are long — the crowd is often wrong at extremes (leans bearish).")
        elif r <= 0.8:
            out.append(f"Long/short accounts {r:.2f}: most traders are short — leans bullish (squeeze risk).")
        else:
            out.append(f"Long/short accounts {r:.2f}: fairly balanced.")
    L = b["liq_24h"]
    tot = L["long_usd"] + L["short_usd"]
    if tot > 0:
        side = "longs" if L["long_usd"] > L["short_usd"] else "shorts"
        out.append(f"Last 24h on OKX: ${L['long_usd']/1e6:.1f}M longs and ${L['short_usd']/1e6:.1f}M shorts liquidated — mostly {side} got wiped.")
    return out


def update(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "derivs.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    res = {"made": datetime.now(timezone.utc).isoformat(timespec="minutes"), "source": "OKX perpetuals"}
    for coin, inst in COINS.items():
        try:
            blk, liqs = coin_block(coin, inst, old.get(coin, {}).get("liqs", []))
            blk["liqs"] = liqs
            res[coin] = blk
        except Exception as e:  # noqa: BLE001
            print(f"derivs {coin} failed: {e}")
            if coin in old:
                res[coin] = old[coin]
    p.write_text(json.dumps(res, separators=(",", ":")))
    for c in COINS:
        if c in res:
            print(c, " | ".join(res[c].get("read", [])))
    return res


if __name__ == "__main__":
    from .paths import ROOT
    update(ROOT / "out" / "trades")
