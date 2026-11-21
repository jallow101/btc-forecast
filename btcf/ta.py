"""Moving averages (7/25/99/200) and RSI (6/12/24) on 15m, 1h, 4h, 1D.

Data: Kraken public OHLC (up to 720 candles per timeframe). Run in GitHub
Actions; writes reports/ta.json and reports/ta.md.
"""
import json
import time
import urllib.request
from datetime import datetime, timezone

from .paths import ROOT

PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "WLD": "WLDUSD"}
TFS = {"15m": 15, "1h": 60, "4h": 240, "1D": 1440}
MAS = [7, 25, 99, 200]
RSIS = [6, 12, 24]


def ohlc(pair, minutes):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={minutes}"
    with urllib.request.urlopen(url, timeout=30) as r:
        js = json.load(r)
    if js.get("error"):
        raise RuntimeError(js["error"])
    key = [k for k in js["result"] if k != "last"][0]
    rows = js["result"][key]
    return [float(x[4]) for x in rows], [float(x[3]) for x in rows], [float(x[2]) for x in rows]


def sma(c, n):
    return sum(c[-n:]) / n if len(c) >= n else None


def rsi(c, n):
    """Wilder RSI, as Binance shows it."""
    if len(c) <= n:
        return None
    g = l = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        l += max(-d, 0)
    g /= n
    l /= n
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        l = (l * (n - 1) + max(-d, 0)) / n
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def run():
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "coins": {}}
    for coin, pair in PAIRS.items():
        out["coins"][coin] = {}
        for tf, m in TFS.items():
            c, lo, hi = ohlc(pair, m)
            out["coins"][coin][tf] = {
                "last": c[-1],
                "low_50": min(lo[-50:]),
                "high_50": max(hi[-50:]),
                "ma": {str(n): sma(c, n) for n in MAS},
                "rsi": {str(n): rsi(c, n) for n in RSIS},
                "n": len(c),
            }
            time.sleep(1.2)
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "ta.json").write_text(json.dumps(out, indent=1))
    lines = [f"# MA / RSI snapshot — {out['generated']}", ""]
    for coin, tfs in out["coins"].items():
        lines += [f"## {coin}", "", "| TF | Last | MA7 | MA25 | MA99 | MA200 | RSI6 | RSI12 | RSI24 |", "|---|---|---|---|---|---|---|---|---|"]
        for tf, d in tfs.items():
            f = lambda v: "—" if v is None else f"{v:,.4f}" if d["last"] < 10 else f"{v:,.2f}"
            r = lambda v: "—" if v is None else f"{v:.1f}"
            lines.append(f"| {tf} | {f(d['last'])} | " + " | ".join(f(d['ma'][str(n)]) for n in MAS)
                         + " | " + " | ".join(r(d['rsi'][str(n)]) for n in RSIS) + " |")
        lines.append("")
    (rep / "ta.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    run()
