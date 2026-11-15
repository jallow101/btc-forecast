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
