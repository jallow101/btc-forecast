"""Weekly BTC dataset + features. Weeks end Saturday. The latest week may be partial (still open):
it is used as the starting point for forecasts but never as a training target."""
import numpy as np
import pandas as pd
from .paths import DAILY_CSV

HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])
HORIZONS = [1, 2, 4, 8, 12, 16, 20, 26, 39, 52, 78, 104, 156, 208]
FEATS = ["d200", "dd", "wk_since_ath", "ph_sin", "ph_cos", "wk_since_halv",
         "mom4", "mom13", "mom26", "mom52", "vol13"]


def load_daily(path=DAILY_CSV):
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return d[["open", "high", "low", "close"]].astype(float)


def weekly(daily):
    w = daily.resample("W-SAT").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    w["complete"] = w.index <= daily.index.max()          # week's Saturday already has a daily close
    w["last_day"] = daily["close"].resample("W-SAT").apply(lambda s: s.index.max()).reindex(w.index)
    return w


def add_features(w):
    w = w.copy()
    w["lp"] = np.log(w["close"])
    w["r1"] = w["lp"].diff()
    sma200 = w["close"].rolling(200, min_periods=100).mean()
    w["sma200"] = sma200
    w["d200"] = w["lp"] - np.log(sma200)
    w["ath"] = w["high"].cummax()
    w["dd"] = w["lp"] - np.log(w["ath"])
    is_ath = w["high"] >= w["ath"]
    w["wk_since_ath"] = w.groupby(is_ath.cumsum()).cumcount().clip(upper=150)

    def weeks_since_halv(t):
        past = HALVINGS[HALVINGS <= t]
        if len(past) == 0:
            return (t - pd.Timestamp("2012-11-28")).days / 7
        return (t - past[-1]).days / 7
    w["wk_since_halv"] = [weeks_since_halv(t) for t in w.index]
    ph = 2 * np.pi * w["wk_since_halv"] / 208.0
    w["ph_sin"], w["ph_cos"] = np.sin(ph), np.cos(ph)
    for k in (4, 13, 26, 52):
        w[f"mom{k}"] = w["lp"].diff(k)
    w["vol13"] = w["r1"].rolling(13).std()
    # targets only from complete weeks
    lp_c = w["lp"].where(w["complete"])
    for h in HORIZONS:
        w[f"y{h}"] = lp_c.shift(-h) - w["lp"]
        w.loc[~w["complete"], f"y{h}"] = np.nan
    return w


def build():
    return add_features(weekly(load_daily()))
