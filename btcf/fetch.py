"""Fetch daily BTC/USD OHLC. Primary: Bitstamp public API (fills only missing days).
Fallback: the public mirror github.com/rcaljouw/btc-price-history."""
import io
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd
from .paths import DAILY_CSV

MIRROR = "https://raw.githubusercontent.com/rcaljouw/btc-price-history/main/data/btc-usd-daily.csv"
BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start={start}"
COLS = ["date", "open", "high", "low", "close", "volume"]


def _get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "btc-forecast/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def from_mirror():
    df = pd.read_csv(io.BytesIO(_get(MIRROR)))
    return df[COLS]


def from_bitstamp(since):
    start = int(datetime(since.year, since.month, since.day, tzinfo=timezone.utc).timestamp())
    rows = json.loads(_get(BITSTAMP.format(start=start)))["data"]["ohlc"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.strftime("%Y-%m-%d")
    return df[COLS]


def update():
    today = datetime.now(timezone.utc).date()
    have = pd.read_csv(DAILY_CSV) if DAILY_CSV.exists() else pd.DataFrame(columns=COLS)
    if have.empty:
        have = from_mirror()
    last = pd.to_datetime(have["date"]).max().date()
    if last < today - timedelta(days=1):
        try:
            new = from_bitstamp(last + timedelta(days=1))
        except Exception as e:  # noqa: BLE001
            print("bitstamp failed, using mirror:", e)
            new = from_mirror()
        have = pd.concat([have, new])
    have["date"] = pd.to_datetime(have["date"]).dt.strftime("%Y-%m-%d")
    have = have.drop_duplicates("date", keep="last").sort_values("date")
    have = have[have["date"] < today.isoformat()]          # only completed UTC days
    have.to_csv(DAILY_CSV, index=False)
    print("daily rows:", len(have), "last:", have["date"].iloc[-1])
    return have


if __name__ == "__main__":
    update()


# ---------- other coins (ETH, SOL): Coinbase Exchange daily candles, Bitstamp as fallback ----------
COINBASE = "https://api.exchange.coinbase.com/products/{pid}/candles?granularity=86400&start={s}&end={e}"
BITSTAMP_PAIR = "https://www.bitstamp.net/api/v2/ohlc/{pair}/?step=86400&limit=1000&start={start}"


def from_coinbase(pid, since):
    """Daily candles from `since` to yesterday, 300 per request."""
    rows, t = [], pd.Timestamp(since)
    end = pd.Timestamp(datetime.now(timezone.utc).date())
    while t < end:
        t1 = min(t + pd.Timedelta(days=299), end)
        url = COINBASE.format(pid=pid, s=t.strftime("%Y-%m-%dT00:00:00Z"), e=t1.strftime("%Y-%m-%dT00:00:00Z"))
        batch = json.loads(_get(url))
        rows += batch
        t = t1 + pd.Timedelta(days=1)
        time.sleep(0.4)
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d")
    return df[COLS]


def from_bitstamp_pair(pair, since):
    rows, t = [], pd.Timestamp(since)
    end = pd.Timestamp(datetime.now(timezone.utc).date())
    while t < end:
        start = int(t.tz_localize("UTC").timestamp())
        data = json.loads(_get(BITSTAMP_PAIR.format(pair=pair, start=start)))["data"]["ohlc"]
        if not data:
            break
        rows += data
        last = pd.to_datetime(int(data[-1]["timestamp"]), unit="s")
        if last <= t:
            break
        t = last + pd.Timedelta(days=1)
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.strftime("%Y-%m-%d")
    return df[COLS]


def update_asset(asset):
    if asset.legacy:
        return update()
    path, today = asset.daily_csv, datetime.now(timezone.utc).date()
    have = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    since = (pd.to_datetime(have["date"]).max() + pd.Timedelta(days=1)) if len(have) else pd.Timestamp("2015-01-01")
    if since.date() < today:
        try:
            new = from_coinbase(asset.coinbase, since)
        except Exception as e:  # noqa: BLE001
            print(f"{asset.key}: Coinbase failed ({e}), trying Bitstamp")
            new = from_bitstamp_pair(asset.bitstamp, since)
        have = pd.concat([have, new])
    have["date"] = pd.to_datetime(have["date"]).dt.strftime("%Y-%m-%d")
    have = have.drop_duplicates("date", keep="last").sort_values("date")
    have = have[have["date"] < today.isoformat()]
    for c in ["open", "high", "low", "close"]:
        have[c] = pd.to_numeric(have[c], errors="coerce")
    have = have.dropna(subset=["close"])
    have.to_csv(path, index=False)
    print(f"{asset.key}: daily rows {len(have)}, {have['date'].iloc[0]} to {have['date'].iloc[-1]}")
    return have
