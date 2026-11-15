"""Final model = average of the three models' 10/50/90% quantiles (6-yr climatology, GBM quantile,
kNN analogs), band widened by split-conformal factors from the walk-forward backtest, then 20,000
Monte Carlo paths (block-bootstrapped weekly shocks) matched to those per-horizon quantiles.
Forecasts every week up to 208 weeks (4 years) ahead."""
import numpy as np
import pandas as pd
from .features import HORIZONS
from .models import baseline_uncond, gbm_fit_predict, knn_analog, halving_analog, powerlaw_fit, powerlaw_predict, pl_weight

STEPS = 208
LEVELS = [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 80000, 90000,
          100000, 120000, 150000, 200000, 250000, 300000]
MIN_CAL = 40   # fewest backtest outcomes needed to calibrate a horizon on its own


POOL_FROM = 104   # 2-4 year horizons have few independent test cycles: calibrate them together
MAX_SCALE = 2.5


def conformal_scales(P):
    E = P[P.model == "ensemble"]
    scale, cover_raw, n = {}, {}, {}
    q80 = lambda v: float(np.quantile(v, 0.8)) if len(v) else 1.0
    for h in HORIZONS:
        d = E[E.h == h]
        n[h] = len(d)
        if len(d):
            cover_raw[h] = float(np.mean((d.y >= d.q10) & (d.y <= d.q90)))
        pool = E[E.h >= POOL_FROM] if h >= POOL_FROM else d
        if len(pool) < MIN_CAL:
            continue
        lo = (pool.q50 - pool.y) / (pool.q50 - pool.q10)
        hi = (pool.y - pool.q50) / (pool.q90 - pool.q50)
        s_lo, s_hi = q80(lo[pool.y < pool.q50]), q80(hi[pool.y >= pool.q50])
        if h >= POOL_FROM and len(d) >= MIN_CAL:          # never narrower than this horizon's own misses
            s_lo = max(s_lo, q80(((d.q50 - d.y) / (d.q50 - d.q10))[d.y < d.q50]))
        scale[h] = (min(max(s_lo, 1.0), MAX_SCALE), min(max(s_hi, 1.0), MAX_SCALE))
    known = sorted(scale)
    for h in HORIZONS:
        if h not in scale:
            prev = [k for k in known if k < h] or known
            scale[h] = (max(scale[k][0] for k in prev), max(scale[k][1] for k in prev))
    return scale, cover_raw, n


def run(w, P, n_paths=20000, seed=42):
    scale, cover_raw, n_cal = conformal_scales(P)
    now = w.index[-1]
    x = w.loc[[now]]
    p0 = float(w.loc[now, "close"])
    plf = powerlaw_fit(w.loc[w["complete"], "lp"])
    rows = []
    for h in HORIZONS:
        train = w[(w.index + pd.Timedelta(weeks=h) <= now) & w["complete"]]
        b = baseline_uncond(train, h)
        g = gbm_fit_predict(train, x, h)[0]
        k, _ = knn_analog(train, x.iloc[0], h)
        hv = halving_analog(w, now, h, now + pd.Timedelta(days=1))
        pl = powerlaw_predict(plf, now, float(w.loc[now, "lp"]), h)
        a = pl_weight(h)
        e = np.sort((1 - a) * (b + g + k) / 3 + a * pl)
        sl, sh = scale[h]
        rows.append(dict(h=h, base=b[1], gbm=g[1], knn=k[1], powerlaw=pl[1], pl_weight=a, halving=(hv[0] if hv else np.nan),
                         q10=e[1] - sl * (e[1] - e[0]), q50=e[1], q90=e[1] + sh * (e[2] - e[1])))
    F = pd.DataFrame(rows).set_index("h")

    H = np.arange(1, STEPS + 1)
    m = np.interp(H, F.index, F.q50)
    dn = np.interp(H, F.index, F.q50 - F.q10) / 1.2816
    up = np.interp(H, F.index, F.q90 - F.q50) / 1.2816
    rng = np.random.default_rng(seed)
    base = w.loc["2014-01-01":]
    r = base.loc[base["complete"], "r1"].dropna().values
    r = (r - r.mean()) / r.std()
    B = 4; nb = int(np.ceil(STEPS / B))
    starts = rng.integers(0, len(r) - B, size=(n_paths, nb))
    shocks = np.concatenate([r[starts[:, i][:, None] + np.arange(B)] for i in range(nb)], axis=1)[:, :STEPS]
    Z = np.cumsum(shocks, axis=1) / np.sqrt(H)
    paths = np.exp(np.log(p0) + m + np.where(Z < 0, Z * dn, Z * up)).astype(np.float32)
    dates = [now + pd.Timedelta(weeks=int(i)) for i in H]
    wk = [d.date().isoformat() for d in dates]

    pct = lambda q: np.percentile(paths, q, axis=0).round(0)
    weekly = pd.DataFrame({"week_ending": wk, "h": H, "p10": pct(10), "p25": pct(25), "median": pct(50),
                           "p75": pct(75), "p90": pct(90)})

    runmin = np.minimum.accumulate(paths, axis=1)                       # lowest close so far, per path
    lowdist = pd.DataFrame({"week_ending": wk,
                            **{f"low_p{q}": np.percentile(runmin, q, axis=0).round(0) for q in (10, 25, 50)}})
    risk = []
    for L in LEVELS:
        touch = (runmin <= L).mean(axis=0) * 100
        above = (paths >= L).mean(axis=0) * 100
        risk += [{"week_ending": wk[i], "level": L, "p_touch": round(float(touch[i]), 1),
                  "p_end_above": round(float(above[i]), 1)} for i in range(STEPS)]
    risk = pd.DataFrame(risk)

    # month of the lowest close, for every window "from now to week k"
    months = np.array([d.strftime("%Y-%m") for d in dates])
    argmin = np.zeros(n_paths, dtype=int)
    lowm = []
    for k in range(STEPS):
        if k > 0:
            argmin = np.where(paths[:, k] < paths[np.arange(n_paths), argmin], k, argmin)
        vc = pd.Series(months[argmin]).value_counts(normalize=True)
        lowm += [{"week_ending": wk[k], "month": mo, "share_pct": round(float(v) * 100, 1)} for mo, v in vc.items() if v >= 0.005]
    lowmonth = pd.DataFrame(lowm)

    comps = pd.DataFrame({"h": F.index, "week_ending": [(now + pd.Timedelta(weeks=int(h))).date().isoformat() for h in F.index],
                          **{c: (p0 * np.exp(F[c])).round(0) for c in ["base", "gbm", "knn", "powerlaw", "halving", "q10", "q50", "q90"]},
                          "pl_weight": F["pl_weight"].values})
    info = {"as_of_week": now.date().isoformat(), "week_complete": bool(w.loc[now, "complete"]),
            "last_day": pd.Timestamp(w.loc[now, "last_day"]).date().isoformat(), "price": round(p0),
            "end_week": wk[-1], "cover_raw": cover_raw, "scale": scale, "n_cal": n_cal}
    return dict(weekly=weekly, risk=risk, lowdist=lowdist, lowmonth=lowmonth, components=comps, info=info)
