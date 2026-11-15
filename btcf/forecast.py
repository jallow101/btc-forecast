"""Final model = average of the three models' 10/50/90% quantiles (6-yr climatology, GBM quantile,
kNN analogs), band widened by split-conformal factors from the walk-forward backtest, then 20,000
Monte Carlo paths with block-bootstrapped weekly shocks matched to those per-horizon quantiles."""
import numpy as np
import pandas as pd
from .features import HORIZONS
from .models import baseline_uncond, gbm_fit_predict, knn_analog, halving_analog

STEPS = 23
LEVELS = [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 80000, 90000, 100000, 120000, 150000]


def conformal_scales(P):
    E = P[P.model == "ensemble"]
    scale, cover_raw = {}, {}
    for h in HORIZONS:
        d = E[E.h == h]
        lo = (d.q50 - d.y) / (d.q50 - d.q10)
        hi = (d.y - d.q50) / (d.q90 - d.q50)
        cover_raw[h] = float(np.mean((d.y >= d.q10) & (d.y <= d.q90)))
        scale[h] = (max(float(np.quantile(lo[d.y < d.q50], 0.8)), 1.0),
                    max(float(np.quantile(hi[d.y >= d.q50], 0.8)), 1.0))
    return scale, cover_raw


def run(w, P, n_paths=20000, seed=42):
    scale, cover_raw = conformal_scales(P)
    now = w.index[-1]
    x = w.loc[[now]]
    p0 = float(w.loc[now, "close"])
    rows = []
    for h in HORIZONS:
        train = w[(w.index + pd.Timedelta(weeks=h) <= now) & w["complete"]]
        b = baseline_uncond(train, h)
        g = gbm_fit_predict(train, x, h)[0]
        k, _ = knn_analog(train, x.iloc[0], h)
        hv = halving_analog(w, now, h, now + pd.Timedelta(days=1))
        e = np.sort((b + g + k) / 3)
        sl, sh = scale[h]
        rows.append(dict(h=h, base=b[1], gbm=g[1], knn=k[1], halving=(hv[0] if hv else np.nan),
                         q10=e[1] - sl * (e[1] - e[0]), q50=e[1], q90=e[1] + sh * (e[2] - e[1])))
    F = pd.DataFrame(rows).set_index("h")

    H = np.arange(1, STEPS + 1)
    m = np.interp(H, F.index, F.q50)
    dn = np.interp(H, F.index, F.q50 - F.q10) / 1.2816
    up = np.interp(H, F.index, F.q90 - F.q50) / 1.2816
    rng = np.random.default_rng(seed)
    r = w.loc["2014-01-01":, "r1"][w.loc["2014-01-01":, "complete"]].dropna().values
    r = (r - r.mean()) / r.std()
    B = 4; nb = int(np.ceil(STEPS / B))
    starts = rng.integers(0, len(r) - B, size=(n_paths, nb))
    shocks = np.concatenate([r[starts[:, i][:, None] + np.arange(B)] for i in range(nb)], axis=1)[:, :STEPS]
    Z = np.cumsum(shocks, axis=1) / np.sqrt(H)
    paths = np.exp(np.log(p0) + m + np.where(Z < 0, Z * dn, Z * up))
    dates = [now + pd.Timedelta(weeks=int(i)) for i in H]

    weekly = pd.DataFrame({"week_ending": [d.date().isoformat() for d in dates], "h": H,
                           **{k: np.percentile(paths, q, axis=0).round(0)
                              for k, q in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))}})
    mins = paths.min(axis=1)
    end = paths[:, -1]
    risk = pd.DataFrame({"level": LEVELS,
                         "p_touch": [round(float(np.mean(mins <= L)) * 100, 1) for L in LEVELS],
                         "p_end_above": [round(float(np.mean(end >= L)) * 100, 1) for L in LEVELS]})
    lowm = pd.Series([dates[i].strftime("%Y-%m") for i in paths.argmin(axis=1)]).value_counts(normalize=True).sort_index()
    comps = pd.DataFrame({"h": F.index, **{c: (p0 * np.exp(F[c])).round(0) for c in ["base", "gbm", "knn", "halving", "q10", "q50", "q90"]}})
    info = {"as_of_week": now.date().isoformat(), "week_complete": bool(w.loc[now, "complete"]),
            "last_day": pd.Timestamp(w.loc[now, "last_day"]).date().isoformat(), "price": round(p0),
            "end_week": dates[-1].date().isoformat(),
            "min_q": {q: round(float(np.percentile(mins, q))) for q in (10, 25, 50, 75, 90)},
            "cover_raw": cover_raw, "scale": scale}
    return dict(weekly=weekly, risk=risk, lowmonth=lowm, components=comps, info=info)
