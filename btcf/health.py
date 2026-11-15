"""Model health: is the model still as accurate as its own long-run record?

Uses the walk-forward store, which is extended every week exactly as the live model would have
forecast it. For each horizon it compares the most recent weeks with the whole history:
  - band hit rate: share of outcomes inside the final 80% band (should be about 80%)
  - typical error: median absolute % miss of the median forecast
Status: OK / WATCH / WARNING. A WARNING means the market is behaving differently from what the model
was calibrated on, and its ranges should not be trusted until it recovers or is reworked."""
import numpy as np
import pandas as pd
from .forecast import band, conformal_scales, recent_vol, vol_z

CHECKS = {1: 13, 4: 13, 12: 26}          # horizon (weeks) -> recent window (weeks of outcomes)
MIN_N = 8


def _grade(cover, ratio):
    if cover < 55 or ratio > 1.75:
        return "WARNING"
    if cover < 65 or ratio > 1.35:
        return "WATCH"
    return "OK"


def check(w, P):
    E = P[P.model == "ensemble"].copy()
    if E.empty:
        return {"status": "N/A", "rows": [], "note": "Not enough history yet."}
    scale, _, _ = conformal_scales(P)
    zv = vol_z(P, w)
    E["v"] = E.t.map(recent_vol(w))
    E["target"] = E.t + pd.to_timedelta(E.h * 7, unit="D")
    last = E.target.max()
    rows = []
    for h, win in CHECKS.items():
        d = E[E.h == h].dropna(subset=["v"])
        if d.empty:
            continue
        b = [band(r.q10, r.q50, r.q90, h, r.v, scale, zv) for r in d.itertuples()]
        d = d.assign(lo=[x[0] for x in b], hi=[x[1] for x in b])
        d = d.assign(inb=(d.y >= d.lo) & (d.y <= d.hi), ape=np.abs(np.exp(d.y - d.q50) - 1) * 100)
        rec = d[d.target > last - pd.Timedelta(weeks=win)]
        if len(rec) < MIN_N:
            continue
        cover, err, err_all = rec.inb.mean() * 100, rec.ape.median(), d.ape.median()
        ratio = err / err_all if err_all else 1.0
        rows.append({"h": h, "window_weeks": win, "n": int(len(rec)), "cover_recent": round(cover, 1),
                     "cover_all": round(d.inb.mean() * 100, 1), "err_recent": round(err, 1), "err_all": round(err_all, 1),
                     "err_ratio": round(ratio, 2), "status": _grade(cover, ratio)})
    order = {"OK": 0, "WATCH": 1, "WARNING": 2}
    status = max((r["status"] for r in rows), key=order.get) if rows else "N/A"
    return {"status": status, "as_of": str(last.date()), "rows": rows}
