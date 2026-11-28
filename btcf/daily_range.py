"""Daily highs and lows, 1-7 days ahead (BTC, ETH, SOL).

Target for horizon k (k = 1 is "tomorrow"):
    hi_k = ln(High of day t+k / Close of day t)
    lo_k = ln(Low  of day t+k / Close of day t)
Both are divided by ATR20 * sqrt(k) so one pooled model can serve all
horizons and all volatility regimes; predictions are scaled back.

Models: gradient-boosted quantile regression (10/50/90) on recent ranges,
realised volatility, returns, distance to MA7/25/99/200, RSI, weekday, and
(optionally) distance and weight of the strongest liquidity cluster above /
below price from the candle liquidity engine run on daily bars.

The 80% band is then calibrated (split-conformal, per horizon and side) on the
last 365 days of out-of-sample errors, the same idea as the weekly model.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import liquidity as L
from .assets import ASSETS
from .paths import DATA, ROOT

QS = (0.1, 0.5, 0.9)
K = range(1, 8)
BASE = ["r1", "r3", "r7", "rng1", "atr5", "atr60r", "vol7r", "vol30r", "upw", "dnw",
        "d_ma7", "d_ma25", "d_ma99", "d_ma200", "rsi6", "rsi14", "k", "dow", "wkend"]
LIQ = ["liq_up_d", "liq_dn_d", "liq_imb"]
LIQ_CFG = {"bin": 0.005, "range": 0.20, "min": 0.01}
LOG = DATA / "daily_log.csv"
LATEST = DATA / "daily_range_latest.json"


def raw_path(key):
    """Uncalibrated walk-forward + live predictions: what the band-width (conformal) factors are fitted on."""
    return DATA / f"daily_raw_{key}.csv"


def update_raw(d, key, p):
    """Add today's raw predictions and fill in outcomes that are now known, so calibration rolls forward daily."""
    rp = raw_path(key)
    raw = pd.read_csv(rp, parse_dates=["date", "target_date"]) if rp.exists() else pd.DataFrame()
    cols = ["date", "k", "target_date", "scale", "close", "y_hi", "y_lo"] + [f"{s}_q{q}" for s in ("hi", "lo") for q in (10, 50, 90)]
    new = p.copy()
    new["y_hi"], new["y_lo"] = np.nan, np.nan
    raw = pd.concat([raw, new[cols]], ignore_index=True).drop_duplicates(["date", "k"], keep="last")
    lc = np.log(d.close)
    miss = raw.y_hi.isna() & raw.target_date.isin(d.index)
    if miss.any():
        t = raw.loc[miss]
        raw.loc[miss, "y_hi"] = (np.log(d.high.reindex(t.target_date).to_numpy()) - lc.reindex(t.date).to_numpy()) / t.scale.to_numpy()
        raw.loc[miss, "y_lo"] = (np.log(d.low.reindex(t.target_date).to_numpy()) - lc.reindex(t.date).to_numpy()) / t.scale.to_numpy()
    raw.sort_values(["date", "k"]).to_csv(rp, index=False)
    return raw


def bt_path(key):
    return DATA / f"daily_bt_{key}.csv"


# ── data & features ────────────────────────────────────────────────────
def load(asset):
    d = pd.read_csv(asset.daily_csv, parse_dates=["date"]).set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return d[d.index < today][["open", "high", "low", "close", "volume"]].astype(float)   # complete days only


def _rsi(c, n):
    x = c.diff()
    g = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    lo = (-x.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / lo)


def liquidity_feats(d):
    """Strongest liquidity cluster above / below each day's close (daily bars)."""
    df = d.reset_index().rename(columns={"date": "t"})
    rows = {}

    def cb(i, eng, swept):
        px, w, _, _ = eng.weights(i)
        c = df.close.iat[i]
        out = [np.nan, np.nan, 0.0]
        if len(px):
            dd = px / c - 1
            best = {}
            for side, m in (("up", (dd >= LIQ_CFG["min"]) & (dd <= LIQ_CFG["range"])),
                            ("dn", (dd <= -LIQ_CFG["min"]) & (dd >= -LIQ_CFG["range"]))):
                if m.any():
                    step = c * LIQ_CFG["bin"]
                    b = np.floor((px[m] - c) / step).astype(int)
                    s = pd.Series(w[m]).groupby(b).sum()
                    k = s.idxmax()
                    best[side] = (abs(np.log((c + (k + .5) * step) / c)), float(s.max()))
            if "up" in best:
                out[0] = best["up"][0]
            if "dn" in best:
                out[1] = best["dn"][0]
            if len(best) == 2:
                out[2] = np.log(best["up"][1] / best["dn"][1])
        rows[i] = out

    L.run_engine(df, cb)
    a = np.array([rows[i] for i in range(len(df))])
    return pd.DataFrame(a, index=d.index, columns=["liq_up_raw", "liq_dn_raw", "liq_imb"])


def features(d, with_liq=True):
    f = pd.DataFrame(index=d.index)
    lc = np.log(d.close)
    r = lc.diff()
    rng = np.log(d.high / d.low)
    f["atr20"] = rng.rolling(20).mean()
    a = f["atr20"]
    f["r1"], f["r3"], f["r7"] = r / a, lc.diff(3) / a, lc.diff(7) / a
    f["rng1"] = rng / a
    f["atr5"] = rng.rolling(5).mean() / a
    f["atr60r"] = rng.rolling(60).mean() / a
    f["vol7r"] = r.rolling(7).std() / a
    f["vol30r"] = r.rolling(30).std() / a
    f["upw"] = np.log(d.high / d.close) / a
    f["dnw"] = np.log(d.close / d.low) / a
    for n in (7, 25, 99, 200):
        f[f"d_ma{n}"] = (lc - np.log(d.close.rolling(n).mean())) / a
    f["rsi6"], f["rsi14"] = _rsi(d.close, 6), _rsi(d.close, 14)
    if with_liq:
        lq = liquidity_feats(d)
        f["liq_up_d"] = lq.liq_up_raw / a
        f["liq_dn_d"] = lq.liq_dn_raw / a
        f["liq_imb"] = lq.liq_imb
    return f


def stack(d, f):
    """One row per (day t, horizon k) with normalised targets."""
    lc = np.log(d.close)
    parts = []
    for k in K:
        p = f.copy()
        p["k"] = k
        tgt_day = d.index.to_series().shift(-k)
        p["target_date"] = tgt_day
        p["dow"] = (d.index + pd.Timedelta(days=k)).dayofweek
        p["wkend"] = (p["dow"] >= 5).astype(int)
        scale = f["atr20"] * np.sqrt(k)
        p["scale"] = scale
        p["y_hi"] = (np.log(d.high.shift(-k)) - lc) / scale
        p["y_lo"] = (np.log(d.low.shift(-k)) - lc) / scale
        p["close"] = d.close
        parts.append(p)
    s = pd.concat(parts)
    s.index.name = "date"
    return s.reset_index()


def _fit(X, y, q):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250, learning_rate=0.05,
                                      max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=1.0,
                                      random_state=0)
    return m.fit(X, y)


def fit_predict(train, test, cols):
    tr = train.dropna(subset=[c for c in cols if c not in LIQ] + ["y_hi", "y_lo"])   # NaN liquidity is allowed
    out = pd.DataFrame(index=test.index)
    for side in ("hi", "lo"):
        for q in QS:
            m = _fit(tr[cols].to_numpy(), tr[f"y_{side}"].to_numpy(), q)
            out[f"{side}_q{int(q * 100)}"] = m.predict(test[cols].to_numpy())
    # quantiles can cross: sort each triple
    for side in ("hi", "lo"):
        c = [f"{side}_q10", f"{side}_q50", f"{side}_q90"]
        out[c] = np.sort(out[c].to_numpy(), axis=1)
    return out


# ── walk-forward backtest ──────────────────────────────────────────────
def walk_forward(asset, start="2019-01-01", refit_days=45, with_liq=True):
    d = load(asset)
    f = features(d, with_liq)
    s = stack(d, f)
    cols = BASE + (LIQ if with_liq else [])
    s = s.dropna(subset=[c for c in cols if c not in LIQ] + ["scale"])
    first = max(pd.Timestamp(start), d.index[0] + pd.Timedelta(days=730))
    res = []
    t = first
    end = d.index[-1]
    while t <= end:
        t1 = t + pd.Timedelta(days=refit_days)
        train = s[(s.target_date < t) & s.y_hi.notna()]          # only outcomes known before t
        test = s[(s.date >= t) & (s.date < t1)]
        if len(test) and len(train) > 2000:
            p = fit_predict(train, test, cols)
            res.append(pd.concat([test[["date", "k", "target_date", "scale", "close", "y_hi", "y_lo"]], p], axis=1))
        t = t1
    return pd.concat(res, ignore_index=True)


def conformal(bt, asof, window=365):
    """Per k and side: factor that widens/narrows q10/q90 around q50 so the band covers 80%."""
    out = {}
    recent = bt[(bt.target_date < asof) & (bt.target_date >= asof - pd.Timedelta(days=window))].dropna(subset=["y_hi"])
    for k in K:
        r = recent[recent.k == k]
        for side in ("hi", "lo"):
            if len(r) < 60:
                out[(k, side)] = 1.0
                continue
            y, m = r[f"y_{side}"], r[f"{side}_q50"]
            dev_up, dev_dn = (r[f"{side}_q90"] - m).clip(lower=1e-6), (m - r[f"{side}_q10"]).clip(lower=1e-6)
            z = np.where(y >= m, (y - m) / dev_up, (m - y) / dev_dn)
            out[(k, side)] = float(np.clip(np.quantile(z, 0.8), 0.5, 2.5))
    return out


def apply_conf(df, scales):
    df = df.copy()
    for side in ("hi", "lo"):
        f = df.apply(lambda r: scales.get((int(r.k), side), 1.0), axis=1)
        m = df[f"{side}_q50"]
        df[f"{side}_q90"] = m + (df[f"{side}_q90"] - m) * f
        df[f"{side}_q10"] = m - (m - df[f"{side}_q10"]) * f
    return df


def calibrate_walk_forward(bt):
    """Conformal scaling using only past errors, re-estimated monthly."""
    out = []
    months = pd.date_range(bt.date.min().normalize(), bt.date.max() + pd.offsets.MonthBegin(1), freq="MS")
    for a, b in zip(months[:-1], months[1:]):
        chunk = bt[(bt.date >= a) & (bt.date < b)]
        if chunk.empty:
            continue
        out.append(apply_conf(chunk, conformal(bt, a)))
    return pd.concat(out, ignore_index=True)


# ── evaluation ─────────────────────────────────────────────────────────
def pinball(y, q, tau):
    e = y - q
    return np.mean(np.maximum(tau * e, (tau - 1) * e))


def baseline(bt, d):
    """Naive: last 60 days' empirical quantiles of the same normalised target (per k)."""
    lc = np.log(d.close)
    rng = np.log(d.high / d.low)
    a = rng.rolling(20).mean()
    out = bt.copy()
    for k in K:
        yh = ((np.log(d.high.shift(-k)) - lc) / (a * np.sqrt(k)))
        yl = ((np.log(d.low.shift(-k)) - lc) / (a * np.sqrt(k)))
        # value at date t uses targets whose outcome day <= t (shift by k), last 60 of them
        for side, y in (("hi", yh), ("lo", yl)):
            known = y.shift(k)
            for q in QS:
                out.loc[out.k == k, f"b_{side}_q{int(q * 100)}"] = out.loc[out.k == k, "date"].map(
                    known.rolling(60, min_periods=30).quantile(q))
    return out


def score(bt, prefix=""):
    r = {}
    b = bt.dropna(subset=["y_hi", "y_lo", f"{prefix}hi_q50"])
    for side in ("hi", "lo"):
        y = b[f"y_{side}"] * b.scale
        q10, q50, q90 = (b[f"{prefix}{side}_q{x}"] * b.scale for x in (10, 50, 90))
        r[side] = {"mae_pct": float(np.mean(np.abs(np.exp(y) - np.exp(q50))) * 100),
                   "cover80": float(np.mean((y >= q10) & (y <= q90)) * 100),
                   "width_pct": float(np.mean(np.exp(q90) - np.exp(q10)) * 100),
                   "pinball": float(np.mean([pinball(y, q, t) for q, t in ((q10, .1), (q50, .5), (q90, .9))]) * 100)}
    return r


def backtest_asset(asset):
    print("walk-forward", asset.key)
    d = load(asset)
    with_liq = walk_forward(asset, with_liq=True)
    no_liq = walk_forward(asset, with_liq=False)
    cal = calibrate_walk_forward(with_liq)
    cal_nl = calibrate_walk_forward(no_liq)
    base = baseline(cal, d)
    cal.to_csv(bt_path(asset.key), index=False)
    with_liq.to_csv(raw_path(asset.key), index=False)
    rep = {"period": [str(cal.date.min().date()), str(cal.date.max().date())], "by_k": {}}
    for k in K:
        m, mn, bb = cal[cal.k == k], cal_nl[cal_nl.k == k], base[base.k == k]
        sm, sn, sb = score(m), score(mn), score(bb, "b_")
        # "yesterday's range" rule for k = 1
        rep["by_k"][k] = {"model": sm, "model_no_liq": sn, "baseline_60d": sb}
        if k == 1:
            dd = d.reindex(m.date)
            half = np.log(dd.high / dd.low).to_numpy() / 2
            yh, yl = (m.y_hi * m.scale).to_numpy(), (m.y_lo * m.scale).to_numpy()
            rep["yesterday_range_mae"] = {
                "hi": float(np.nanmean(np.abs(np.exp(yh) - np.exp(half))) * 100),
                "lo": float(np.nanmean(np.abs(np.exp(yl) - np.exp(-half))) * 100)}
    recent = cal[cal.date >= cal.date.max() - pd.Timedelta(days=365)]
    rep["last_365d_k1"] = score(recent[recent.k == 1])
    return rep


def run_backtests():
    out = {}
    for key in ("btc", "eth", "sol"):
        out[key] = backtest_asset(ASSETS[key])
        print(json.dumps(out[key]["by_k"]["1"] if "1" in out[key]["by_k"] else out[key]["by_k"][1], indent=1))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "daily_range_backtest.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ── live: predict, log, evaluate ───────────────────────────────────────
def predict_live(asset):
    d = load(asset)
    f = features(d, True)
    s = stack(d, f)
    cols = BASE + LIQ
    last = d.index[-1]
    train = s.dropna(subset=["y_hi", "y_lo"])
    train = train[train.date >= last - pd.Timedelta(days=3000)]     # last ~8 years: faster, and today's market
    test = s[s.date == last].copy()
    test["target_date"] = [last + pd.Timedelta(days=int(k)) for k in test.k]
    p = pd.concat([test[["date", "k", "target_date", "scale", "close"]], fit_predict(train, test, cols)], axis=1)
    # calibrate band widths on RAW past predictions (the stored backtest is already calibrated,
    # so fitting on it gave ~1.0 factors and left live bands ~5-15% too narrow)
    raw = update_raw(d, asset.key, p)
    p = apply_conf(p, conformal(raw, last + pd.Timedelta(days=1)))
    rows = []
    for r in p.itertuples():
        row = {"asset": asset.key, "run_date": str(last.date()), "target_date": str(r.target_date.date()),
               "k": int(r.k), "close": round(r.close, 4)}
        for side in ("hi", "lo"):
            for q in (10, 50, 90):
                row[f"{side}_q{q}"] = round(float(r.close * np.exp(getattr(r, f"{side}_q{q}") * r.scale)), 4)
        rows.append(row)
    return rows


def evaluate(log, asset):
    d = load(asset)
    lg = log[log.asset == asset.key].copy()
    lg["target_date"] = pd.to_datetime(lg.target_date)
    lg = lg[lg.target_date.isin(d.index)]
    if lg.empty:
        return []
    act = d.reindex(lg.target_date)
    lg["act_hi"], lg["act_lo"] = act.high.to_numpy(), act.low.to_numpy()
    lg["hi_in"] = (lg.act_hi >= lg.hi_q10) & (lg.act_hi <= lg.hi_q90)
    lg["lo_in"] = (lg.act_lo >= lg.lo_q10) & (lg.act_lo <= lg.lo_q90)
    lg["hi_err_pct"] = (lg.act_hi / lg.hi_q50 - 1) * 100
    lg["lo_err_pct"] = (lg.act_lo / lg.lo_q50 - 1) * 100
    lg["target_date"] = lg.target_date.dt.strftime("%Y-%m-%d")
    return lg.round(4).to_dict("records")


def run_live(site_root=None):
    site_root = site_root or (ROOT / "docs" / "data")
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    latest, bt = {}, {}
    rep_p = ROOT / "reports" / "daily_range_backtest.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    for key in ("btc", "eth", "sol"):
        a = ASSETS[key]
        try:
            rows = predict_live(a)
        except Exception as e:  # noqa: BLE001
            print("daily range failed", key, e)
            continue
        new = pd.DataFrame(rows)
        if not log.empty:
            log = log[~((log.asset == key) & (log.run_date == rows[0]["run_date"]))]
        log = pd.concat([log, new], ignore_index=True)
        latest[key] = rows
    log.to_csv(LOG, index=False)
    for key in latest:
        a = ASSETS[key]
        scored = evaluate(log, a)
        k1 = [r for r in scored if r["k"] == 1]
        live_stats = {"n": len(k1),
                      "hi_cover": float(np.mean([r["hi_in"] for r in k1]) * 100) if k1 else None,
                      "lo_cover": float(np.mean([r["lo_in"] for r in k1]) * 100) if k1 else None,
                      "hi_mae": float(np.mean([abs(r["hi_err_pct"]) for r in k1])) if k1 else None,
                      "lo_mae": float(np.mean([abs(r["lo_err_pct"]) for r in k1])) if k1 else None}
        dd = load(a).tail(30)
        recent = [[str(i.date()), r.open, r.high, r.low, r.close] for i, r in dd.iterrows()]
        out = {"asset": key, "made": datetime.now(timezone.utc).isoformat(), "pred": latest[key], "recent": recent,
               "scored": k1[-60:], "live": live_stats, "backtest": rep.get(key)}
        target = a.site_dir if key != "btc" else site_root
        target.mkdir(parents=True, exist_ok=True)
        (target / "daily_range.json").write_text(json.dumps(out, default=float))
    LATEST.write_text(json.dumps({k: v for k, v in latest.items()}, indent=1))
    print(json.dumps({k: v[0] for k, v in latest.items()}, indent=1))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        run_backtests()
    else:
        run_live()
