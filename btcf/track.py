"""Forecast log + live tracking. Every run appends its weekly forecasts to data/forecast_log.csv.
When a target week has closed, its actual close is compared with every forecast made for it."""
import numpy as np
import pandas as pd
from .paths import FORECAST_LOG


def log_run(fc, run_date):
    info = fc["info"]
    rows = fc["weekly"].copy()
    rows.insert(0, "run_date", run_date)
    rows.insert(1, "base_week", info["as_of_week"])
    rows.insert(2, "price_at_run", info["price"])
    old = pd.read_csv(FORECAST_LOG) if FORECAST_LOG.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old.run_date != run_date]               # re-runs on the same day replace that day
    log = pd.concat([old, rows], ignore_index=True)
    log.to_csv(FORECAST_LOG, index=False)
    return log


def evaluate(log, w):
    """One forecast per (target week, horizon): the latest run made that many weeks before the target.
    Horizon is counted from the run's own week, so a mid-week run counts as that week's forecast."""
    if log.empty:
        return pd.DataFrame()
    L = log.copy()
    L["target"] = pd.to_datetime(L.week_ending)
    L["base"] = pd.to_datetime(L.base_week)
    L["hz"] = ((L.target - L.base).dt.days // 7).astype(int)
    L = L.sort_values("run_date").groupby(["week_ending", "hz"]).tail(1)
    actual = w.loc[w.complete, "close"]
    L["actual"] = L.target.map(actual)
    L = L.dropna(subset=["actual"])
    L["err_pct"] = (L["median"] / L["actual"] - 1) * 100
    L["inside80"] = (L.actual >= L.p10) & (L.actual <= L.p90)
    return L[["week_ending", "hz", "run_date", "price_at_run", "p10", "median", "p90", "actual", "err_pct", "inside80"]]
