"""Walk-forward backtest that mirrors the live model: the forecast made in week t uses only data
known by week t (targets that had already closed, prices up to t), with models refit every
`refit_weeks` weeks. Past forecasts never change, so results are stored and only new weeks are
added on each run (data/backtest_preds.csv)."""
import os

import numpy as np
import pandas as pd
from .features import FEATS, HORIZONS
from .models import baseline_uncond, gbm_fit_predict, knn_analog, halving_analog, powerlaw_fit, powerlaw_predict, pl_weight
from .paths import BACKTEST_CSV

FIRST = pd.Timestamp("2017-01-01")
REFIT_WEEKS = 4          # history rebuilt with a refit every 4 weeks; new weeks are added (and fit) one at a time
MIN_TRAIN = 150


def _block(args):
    """Forecasts for one block of forecast weeks, all sharing one model fit made at the block start."""
    w, h, t0, idx = args
    known_prices = w[(w.index <= t0) & w["complete"]]
    plf = powerlaw_fit(known_prices["lp"])
    train = w[(w.index + pd.Timedelta(weeks=h) <= t0) & w["complete"]]
    test = w.loc[idx]
    y = test[f"y{h}"].values
    pl = np.vstack([powerlaw_predict(plf, t, test.loc[t, "lp"], h) for t in test.index])
    a = pl_weight(h)
    out = []
    if len(train.dropna(subset=FEATS + [f"y{h}"])) >= MIN_TRAIN:
        b = np.tile(baseline_uncond(train, h), (len(test), 1))
        g = gbm_fit_predict(train, test, h)
        k = np.vstack([knn_analog(train, test.loc[t], h)[0] for t in test.index])
        e = np.sort((1 - a) * (b + g + k) / 3 + a * pl, axis=1)
        models = (("baseline", b), ("gbm", g), ("knn", k), ("powerlaw", pl), ("ensemble", e))
    else:
        models = (("powerlaw", pl), ("ensemble", np.sort(pl, axis=1)))
    for name, P in models:
        out.append(pd.DataFrame({"t": test.index, "h": h, "model": name, "y": y,
                                 "q10": P[:, 0], "q50": P[:, 1], "q90": P[:, 2]}))
    hv = [(lambda r: r[0] if r else np.nan)(halving_analog(w, t, h, t0 + pd.Timedelta(days=1))) for t in test.index]
    out.append(pd.DataFrame({"t": test.index, "h": h, "model": "halving", "y": y, "q10": np.nan, "q50": hv, "q90": np.nan}))
    return pd.concat(out, ignore_index=True)


def run(w, refit_weeks=REFIT_WEEKS, store=BACKTEST_CSV):
    old = pd.read_csv(store, parse_dates=["t"]) if os.path.exists(store) else pd.DataFrame(columns=["t", "h"])
    if len(old) and "refit" in old.columns and old["refit"].iloc[0] != refit_weeks:
        old = pd.DataFrame(columns=["t", "h"])               # stored with another cadence: rebuild
    done = set(zip(old["t"], old["h"])) if len(old) else set()
    tasks = []
    for h in HORIZONS:
        ok = w[(w.index >= FIRST) & w["complete"]].dropna(subset=FEATS + [f"y{h}"])
        todo = [t for t in ok.index if (t, h) not in done]
        for i in range(0, len(todo), refit_weeks):
            blk = todo[i:i + refit_weeks]
            tasks.append((w, h, blk[0], blk))
    if tasks:
        # sequential on purpose: the gradient-boosting fits already use every core, and worker
        # processes on top of that fight over threads and run many times slower
        new = pd.concat([_block(t) for t in tasks], ignore_index=True)
        new["refit"] = refit_weeks
        P = pd.concat([old, new], ignore_index=True) if len(old) else new
    else:
        P = old
    P = P.dropna(subset=["q50"]).sort_values(["h", "t", "model"]).reset_index(drop=True)
    P.to_csv(store, index=False, float_format="%.5f")
    print(f"backtest: {len(tasks)} new model fits, {len(P)} stored forecasts")
    return P
