"""Long 15-minute history for the day-trade backtest (OKX spot, free, no key).

Hyperliquid only gives the last 5000 candles (~52 days of 15m), which is one
market regime. OKX's history endpoint pages back years, 100 candles a call.
Writes data/hist/<coin>_15m.csv (same columns as data/liq). Runs on the
GitHub runner; incremental after the first run.
"""
import csv
import json
import time
import urllib.request

from .paths import DATA

COINS = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "SOL": "SOL-USDT"}
OUT = DATA / "hist"
DAYS = 400
UA = {"User-Agent": "btc-forecast/1.0"}


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


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


def _page(inst, have, after, until):
    """Page backwards from `after` (ms, '' = now) until candles older than `until` (s)."""
    calls = 0
    while True:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=15m&limit=100" + (
            f"&after={after}" if after else "")
        rows = _get(url).get("data", [])
        if not rows:
            break
        for r in rows:
            ts = int(r[0]) // 1000
            if r[8] == "1":         # confirmed candles only
                have[ts] = (ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
        after = rows[-1][0]
        calls += 1
        if int(after) // 1000 <= until or calls > 600:
            break
        time.sleep(0.12)
    return calls


def fetch(coin, inst):
    path = OUT / f"{coin}_15m.csv"
    have = load(path)
    stop = int(time.time()) - DAYS * 86400
    calls = 0
    if have:
        calls += _page(inst, have, "", max(have))              # new candles since last run
        if min(have) > stop + 86400:                             # extend further back if needed
            calls += _page(inst, have, str(min(have) * 1000), stop)
    else:
        calls += _page(inst, have, "", stop)
    have = {k: v for k, v in have.items() if k >= stop}
    save(path, have)
    print(f"{coin}: {len(have)} bars, {calls} calls")


def update():
    OUT.mkdir(parents=True, exist_ok=True)
    for coin, inst in COINS.items():
        try:
            fetch(coin, inst)
        except Exception as e:  # noqa: BLE001
            print(f"{coin}: OKX history failed: {e}")


if __name__ == "__main__":
    update()
