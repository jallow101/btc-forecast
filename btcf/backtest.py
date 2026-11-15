"""Walk-forward backtest: for each test year Y, every model is trained only on weeks whose outcome
was known before 1 January of Y, then forecasts every week of Y."""
import numpy as np
import pandas as pd
from .features import FEATS, HORIZONS
from .models import baseline_uncond, gbm_fit_predict, knn_analog, halving_analog, powerlaw_fit, powerlaw_predict, pl_weight
from .paths import BACKTEST_CSV


def run(w, first_year=2017):
    last_year = int(w.index[-1].year)
    preds = []
    for Y in range(first_year, last_year + 1):
        ys = pd.Timestamp(f"{Y}-01-01"); ye = pd.Timestamp(f"{Y+1}-01-01")
        known = w[(w.index < ys) & w["complete"]]
        plf = powerlaw_fit(known["lp"])                       # prices known before the test year only
        for h in HORIZONS:
            test = w[(w.index >= ys) & (w.index < ye)].dropna(subset=FEATS + [f"y{h}"])
            if len(test) == 0:
                continue
            train = w[w.index + pd.Timedelta(weeks=h) < ys]
            y = test[f"y{h}"].values
            pl = np.vstack([powerlaw_predict(plf, t, test.loc[t, "lp"], h) for t in test.index])
            a = pl_weight(h)
            if len(train.dropna(subset=FEATS + [f"y{h}"])) >= 150:
                b = np.tile(baseline_uncond(train, h), (len(test), 1))
                g = gbm_fit_predict(train, test, h)
                k = np.vstack([knn_analog(train, test.loc[t], h)[0] for t in test.index])
                e = np.sort((1 - a) * (b + g + k) / 3 + a * pl, axis=1)
            else:                                              # long horizons early on: trend model only
                b = g = k = np.full((len(test), 3), np.nan)
                e = np.sort(pl, axis=1)
            hv = np.array([(lambda r: r[0] if r else np.nan)(halving_analog(w, t, h, ys)) for t in test.index])
            for name, P in (("baseline", b), ("gbm", g), ("knn", k), ("powerlaw", pl), ("ensemble", e)):
                preds.append(pd.DataFrame({"t": test.index, "h": h, "model": name, "y": y,
                                           "q10": P[:, 0], "q50": P[:, 1], "q90": P[:, 2]}))
            preds.append(pd.DataFrame({"t": test.index, "h": h, "model": "halving", "y": y,
                                       "q10": np.nan, "q50": hv, "q90": np.nan}))
    P = pd.concat(preds, ignore_index=True)
    P = P.dropna(subset=["q50"]).reset_index(drop=True) if "q50" in P else P
    P.to_csv(BACKTEST_CSV, index=False)
    return P
