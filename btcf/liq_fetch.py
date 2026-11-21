"""Candle history for the liquidity module.

Primary: Hyperliquid perps (perp volume is the best free proxy for leveraged
positioning; up to 5000 candles per interval). Fallback: Kraken spot (720).
Files: data/liq/<coin>_<tf>.csv, merged with what is already there so
history grows with every run.
"""
import csv
import json
import time
import urllib.request

from .paths import DATA

COINS = ["BTC", "ETH", "SOL", "WLD"]
TFS = {"15m": 15, "1h": 60, "4h": 240}
KRAKEN = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "WLD": "WLDUSD"}
OUT = DATA / "liq"
UA = {"User-Agent": "btc-forecast/1.0", "Content-Type": "application/json"}


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def hyperliquid(coin, tf, minutes):
    end = int(time.time() * 1000)
    start = end - 5000 * minutes * 60_000
    rows = _post("https://api.hyperliquid.xyz/info",
                 {"type": "candleSnapshot", "req": {"coin": coin, "interval": tf,
                                                    "startTime": start, "endTime": end}})
    return [(int(r["t"]) // 1000, float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"]))
            for r in rows]


def kraken(coin, minutes):
    js = _get(f"https://api.kraken.com/0/public/OHLC?pair={KRAKEN[coin]}&interval={minutes}")
    key = [k for k in js["result"] if k != "last"][0]
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6]))
            for r in js["result"][key]]


def load(path):
    if not path.exists():
        return {}
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return {int(r[0]): (int(r[0]), *map(float, r[1:6])) for r in rd}


def save(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for r in sorted(rows.values()):
            w.writerow(r)


def update():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for coin in COINS:
        for tf, m in TFS.items():
            path = OUT / f"{coin}_{tf}.csv"
            have = load(path)
            src = "hyperliquid"
            try:
                new = hyperliquid(coin, tf, m)
                if not new:
                    raise RuntimeError("empty")
            except Exception as e:  # noqa: BLE001
                print(f"{coin} {tf}: hyperliquid failed ({e}); using kraken")
                src = "kraken"
                new = kraken(coin, m)
            # the newest candle is still forming: always overwrite it
            for r in new:
                have[r[0]] = r
            save(path, have)
            report[f"{coin}_{tf}"] = {"source": src, "bars": len(have)}
            time.sleep(0.6)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    update()
