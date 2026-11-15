"""Forecasting models. All predict quantiles of h-week forward log return."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import FEATS, HORIZONS, HALVINGS

QS = [0.1, 0.5, 0.9]


def baseline_uncond(train, h):
    """Climatology: empirical quantiles of h-week returns over the last 6 years of training data
    (rolling window chosen in backtest: removes the 2011-2016 hyper-growth drift)."""
    y = train[f"y{h}"].dropna().iloc[-312:]
    return np.quantile(y, QS)


def gbm_fit_predict(train, Xtest, h):
    tr = train.dropna(subset=FEATS + [f"y{h}"])
    out = []
    for q in QS:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200,
                                          learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=0)
        m.fit(tr[FEATS], tr[f"y{h}"])
        out.append(m.predict(Xtest[FEATS]))
    out = np.sort(np.vstack(out).T, axis=1)  # enforce monotone quantiles
    return out


KNN_FEATS = ["d200", "dd", "mom13", "mom52", "vol13"]


def knn_analog(train, xrow, h, k=40):
    """Nearest-neighbour analog: quantiles of the h-week outcomes of the k most similar past weeks."""
    tr = train.dropna(subset=KNN_FEATS + [f"y{h}"])
    mu, sd = tr[KNN_FEATS].mean(), tr[KNN_FEATS].std()
    Z = (tr[KNN_FEATS] - mu) / sd
    z = (xrow[KNN_FEATS].astype(float) - mu) / sd
    dist = np.sqrt(((Z - z) ** 2).sum(axis=1))
    idx = dist.nsmallest(k).index
    return np.quantile(tr.loc[idx, f"y{h}"], QS), idx


def halving_analog(full, t, h, train_end):
    """Cycle analog: returns over the same halving-phase window in each prior complete cycle."""
    past_h = HALVINGS[HALVINGS < t]
    if len(past_h) < 2:
        return None
    phase_w = (t - past_h[-1]).days / 7
    rets = []
    for hv in past_h[:-1]:
        t0 = (hv + pd.Timedelta(weeks=phase_w)).floor("D")
        t1 = (t0 + pd.Timedelta(weeks=h)).floor("D")
        if t1 >= train_end:
            continue
        s = full["lp"]
        p0 = s.asof(t0); p1 = s.asof(t1)
        if np.isfinite(p0) and np.isfinite(p1):
            rets.append(p1 - p0)
    if not rets:
        return None
    rets = np.array(rets)
    med = np.median(rets)
    return med, rets


# ---- Long-run power-law trend (diminishing returns), fit on prices only ----
GENESIS = pd.Timestamp("2009-01-03")


def powerlaw_fit(lp, genesis=GENESIS):
    """log price = a + b*log(days since genesis); residual follows AR(1) at weekly steps."""
    genesis = pd.Timestamp(genesis)
    t = np.log(((lp.index - genesis).days).values.astype(float))
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, lp.values, rcond=None)
    res = lp.values - A @ coef
    phi = float(np.corrcoef(res[:-1], res[1:])[0, 1])
    return coef, res, phi, genesis


def powerlaw_predict(fit, t, lp_t, h):
    """Quantiles (10/50/90) of the h-week log return from week t."""
    coef, res, phi, genesis = fit
    trend = lambda d: coef[0] + coef[1] * np.log((d - genesis).days)
    r0 = lp_t - trend(t)
    med = trend(t + pd.Timedelta(weeks=h)) + (phi ** h) * r0
    ch = res[h:] - (phi ** h) * res[:-h]
    q = np.quantile(ch, QS) if len(ch) > 20 else np.array([-1.0, 0.0, 1.0])
    return med + q - q[1] - lp_t


def pl_weight(h):
    """Share of the power-law model in the final blend: 0 for the next weeks, all of it from 26 weeks."""
    return float(np.interp(h, [4, 8, 12, 20, 26], [0.0, 0.25, 0.5, 0.85, 1.0]))
