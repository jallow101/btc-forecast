"""Write the JSON files the web page (docs/index.html) reads."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from .paths import SITE_DATA

NAMES = {"baseline": "Baseline", "gbm": "Gradient boosting", "knn": "Analogs", "powerlaw": "Power-law trend", "ensemble": "Final model"}
QS = [0.1, 0.5, 0.9]
BT_H = [1, 4, 12, 26, 52, 104, 208]


def _dump(name, obj):
    with open(SITE_DATA / f"{name}.json", "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))


def _records(df):
    return json.loads(df.to_json(orient="records"))


def _pin(d):
    s = 0
    for q, c in zip(QS, ["q10", "q50", "q90"]):
        e = d.y - d[c]
        s += np.mean(np.maximum(q * e, (q - 1) * e))
    return s / 3


def write(w, P, fc, track):
    info = fc["info"]
    scale = info["scale"]
    # history (weekly closes since 2024)
    h = w.loc["2024-01-01":, ["close", "complete"]]
    _dump("history", [{"week_ending": i.date().isoformat(), "close": round(r.close), "complete": bool(r.complete)} for i, r in h.iterrows()])
    _dump("forecast", _records(fc["weekly"]))
    _dump("risk", _records(fc["risk"]))
    _dump("lowmonth", _records(fc["lowmonth"]))
    _dump("lowdist", _records(fc["lowdist"]))
    _dump("components", _records(fc["components"]))

    # backtest paths: forecast made h weeks earlier vs actual, as prices
    B = P[P.model.isin(NAMES) & P.h.isin(BT_H)].copy()
    B["made_close"] = B.t.map(w.close)
    B["target"] = B.t + pd.to_timedelta(B.h * 7, unit="D")
    B["actual"] = B.target.map(w.close)
    B = B.dropna(subset=["actual", "made_close"])
    # calibrated bands for the ensemble (same widening as the live forecast)
    lo_s = B.h.map(lambda k: scale[k][0]); hi_s = B.h.map(lambda k: scale[k][1])
    ens = B.model == "ensemble"
    B["c10"] = np.where(ens, B.q50 - lo_s * (B.q50 - B.q10), B.q10)
    B["c90"] = np.where(ens, B.q50 + hi_s * (B.q90 - B.q50), B.q90)
    bt = pd.DataFrame({"t": B.target.dt.strftime("%Y-%m-%d"), "h": B.h.astype(int), "m": B.model.map(NAMES),
                       "a": B.actual.round(0), "p": (B.made_close * np.exp(B.q50)).round(0),
                       "lo": (B.made_close * np.exp(B.c10)).round(0), "hi": (B.made_close * np.exp(B.c90)).round(0)})
    _dump("bt_paths", _records(bt))

    # error by year
    B["yr"] = B.target.dt.year
    B["ape"] = np.abs(B.made_close * np.exp(B.q50) / B.actual - 1) * 100
    yrs = B.groupby(["yr", "h", "model"]).agg(n=("ape", "size"), err=("ape", "median")).reset_index()
    yrs = yrs[yrs.n >= 8]
    _dump("bt_years", [{"year": int(r.yr), "h": int(r.h), "model": NAMES[r.model], "err": round(r.err, 1)} for r in yrs.itertuples()])

    # scorecard
    P = P.copy(); P["dd"] = P.t.map(w.dd)
    slices = {"Bear-market weeks (like now)": P.dd < np.log(0.6), "All weeks 2017–now": P.t.dt.year > 0, "Recent (2024–now)": P.t.dt.year >= 2024}
    score = []
    for sl, mask in slices.items():
        for hz in BT_H:
            for m in ["baseline", "gbm", "knn", "powerlaw", "ensemble"]:
                d = P[mask & (P.model == m) & (P.h == hz)].dropna(subset=["q50"])
                if len(d) == 0:
                    continue
                score.append({"slice": sl, "h": hz, "model": NAMES[m], "n": len(d), "pinball": round(float(_pin(d)), 4),
                              "err": round(float(np.median(np.abs(np.exp(d.y - d.q50) - 1))) * 100, 1),
                              "dir": round(float(np.mean(np.sign(d.y) == np.sign(d.q50))) * 100, 1),
                              "cover": round(float(np.mean((d.y >= d.q10) & (d.y <= d.q90))) * 100, 1)})
    _dump("scorecard", score)

    # live tracking (forecasts logged by this pipeline, scored once the week has closed)
    live = [] if track is None or track.empty else [
        {"t": r.week_ending, "h": int(r.hz), "run": r.run_date, "a": round(r.actual), "p": round(r.median),
         "lo": round(r.p10), "hi": round(r.p90), "err": round(r.err_pct, 1), "in80": bool(r.inside80)} for r in track.itertuples()]
    _dump("live", live)

    info = {k: v for k, v in info.items() if k not in ("scale", "cover_raw", "n_cal")}
    meta = dict(info, generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                live_scored=len(live), scale={str(k): v for k, v in scale.items()},
                cover_raw={str(k): v for k, v in fc["info"]["cover_raw"].items()},
                n_cal={str(k): v for k, v in fc["info"]["n_cal"].items()})
    _dump("meta", meta)
