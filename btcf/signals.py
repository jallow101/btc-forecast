"""Market signals beyond price, as weekly features (weeks end Saturday).
  fng        Crypto Fear & Greed index (alternative.me), weekly mean, and its 4-week change
  funding    BTC perpetual funding rate (BitMEX XBTUSD), weekly mean, annualised
  dxy        Broad US dollar index (FRED DTWEXBGS), 13-week log change
  ry10       10-year real Treasury yield (FRED DFII10), level and 13-week change
  stable     Total USD stablecoin supply (DefiLlama), 13-week log growth
Raw series are cached in data/signals/*.csv; only missing history is fetched again."""
import io
import json
import time
import urllib.request

import numpy as np
import pandas as pd
from .paths import DATA

DIR = DATA / "signals"
DIR.mkdir(exist_ok=True)
GROUPS = {"fng": ["fng", "fng_chg4"], "funding": ["funding"], "dxy": ["dxy_chg13"],
          "ry10": ["ry10", "ry10_chg13"], "stable": ["stable_g13"]}


def _get(url, tries=3, timeout=120):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(4 * (i + 1))


def fetch_fng():
    d = json.loads(_get("https://api.alternative.me/fng/?limit=0&format=json"))["data"]
    s = pd.Series({pd.to_datetime(int(x["timestamp"]), unit="s").normalize(): float(x["value"]) for x in d})
    return s.sort_index().rename("fng")


def fetch_funding_deribit():
    """Deribit BTC-PERPETUAL 8h funding, fetched in 30-day windows since May 2019."""
    rows, t = [], pd.Timestamp("2019-05-01")
    end = pd.Timestamp.now("UTC").tz_localize(None)
    while t < end:
        t1 = min(t + pd.Timedelta(days=30), end)
        url = ("https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name=BTC-PERPETUAL"
               f"&start_timestamp={int(t.timestamp() * 1000)}&end_timestamp={int(t1.timestamp() * 1000)}")
        res = json.loads(_get(url)).get("result", [])
        rows += [(pd.to_datetime(r["timestamp"], unit="ms"), float(r["interest_8h"])) for r in res]
        t = t1
        time.sleep(0.3)
    s = pd.Series(dict(rows), dtype=float).sort_index()
    s.rename("funding").rename_axis("date").to_csv(DIR / "funding.csv")
    return s.rename("funding")


def fetch_funding():
    try:
        return fetch_funding_bitmex()
    except Exception as e:  # noqa: BLE001
        print("BitMEX funding failed, trying Deribit:", e)
        return fetch_funding_deribit()


def fetch_funding_bitmex():
    path = DIR / "funding.csv"
    old = pd.read_csv(path, parse_dates=["date"]).set_index("date")["funding"] if path.exists() else pd.Series(dtype=float)
    start = (old.index.max() + pd.Timedelta(hours=1)) if len(old) else pd.Timestamp("2016-05-14")
    rows = []
    while True:
        url = ("https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&reverse=false&startTime="
               + start.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        batch = json.loads(_get(url))
        if not batch:
            break
        rows += [(pd.to_datetime(b["timestamp"]).tz_localize(None), float(b["fundingRate"])) for b in batch]
        start = rows[-1][0] + pd.Timedelta(minutes=1)
        if len(batch) < 500:
            break
        time.sleep(1.2)
    new = pd.Series(dict(rows), dtype=float)
    s = pd.concat([old, new]).sort_index()
    s = s[~s.index.duplicated()]
    s.rename("funding").rename_axis("date").to_csv(path)
    return s.rename("funding")


def fetch_fred(series):
    raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2014-01-01", timeout=180)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    return pd.to_numeric(df.set_index("date")["v"], errors="coerce").dropna()


def fetch_stable():
    d = json.loads(_get("https://stablecoins.llama.fi/stablecoincharts/all"))
    s = pd.Series({pd.to_datetime(int(x["date"]), unit="s").normalize(): float(x["totalCirculatingUSD"].get("peggedUSD", np.nan)) for x in d})
    return s.sort_index().dropna().rename("stable")


def update():
    """Fetch every source (each one may fail without stopping the others) and cache raw daily series."""
    fetchers = {"fng": fetch_fng, "funding": fetch_funding, "dxy": lambda: fetch_fred("DTWEXBGS"),
                "ry10": lambda: fetch_fred("DFII10"), "stable": fetch_stable}
    status = {}
    for name, fn in fetchers.items():
        try:
            s = fn()
            if name != "funding":
                s.rename(name).rename_axis("date").to_csv(DIR / f"{name}.csv")
            status[name] = f"ok, {len(s)} rows to {s.index.max().date()}"
        except Exception as e:  # noqa: BLE001
            status[name] = f"failed: {e}"
    return status


def _load(name):
    p = DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").iloc[:, 0]


def weekly_features(index):
    """Signal features aligned to the price model's weekly index (weeks ending Saturday)."""
    out = pd.DataFrame(index=index)
    wk = lambda s, how="mean": getattr(s.resample("W-SAT"), how)().reindex(index)
    fng = _load("fng")
    if fng is not None:
        f = wk(fng)
        out["fng"], out["fng_chg4"] = f, f.diff(4)
    fu = _load("funding")
    if fu is not None:
        out["funding"] = wk(fu) * 3 * 365          # 8-hourly rate -> annualised
    dxy = _load("dxy")
    if dxy is not None:
        out["dxy_chg13"] = np.log(wk(dxy, "last")).diff(13)
    ry = _load("ry10")
    if ry is not None:
        r = wk(ry, "last")
        out["ry10"], out["ry10_chg13"] = r, r.diff(13)
    st = _load("stable")
    if st is not None:
        out["stable_g13"] = np.log(wk(st, "last")).diff(13)
    return out
