"""The coins the pipeline forecasts, and where each one's files live.
BTC keeps its original file names so its forecast log and backtest history carry on unchanged."""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from .paths import DATA, SITE_DATA


@dataclass
class Asset:
    key: str
    name: str
    genesis: str                     # start of the power-law trend clock
    coinbase: str                    # Coinbase Exchange product id
    bitstamp: str                    # Bitstamp pair (fallback source)
    levels: list                     # price levels for the odds tables
    card_level: float                # level shown on the "chance of a weekly close at/below" card
    lede_level: float                # level used in the page's opening sentence
    sma_min: int = 100               # weeks of history before the 200-week average is used
    legacy: bool = False             # BTC: original file names
    notes: str = ""

    @property
    def dir(self) -> Path:
        d = DATA if self.legacy else DATA / self.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def daily_csv(self):
        return self.dir / ("btc-usd-daily.csv" if self.legacy else "daily.csv")

    @property
    def backtest_csv(self):
        return self.dir / "backtest_preds.csv"

    @property
    def forecast_log(self):
        return self.dir / "forecast_log.csv"

    @property
    def site_dir(self):
        d = SITE_DATA if self.legacy else SITE_DATA / self.key
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def genesis_ts(self):
        return pd.Timestamp(self.genesis)


ASSETS = {
    "btc": Asset("btc", "Bitcoin", "2009-01-03", "BTC-USD", "btcusd",
                 [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 80000, 90000,
                  100000, 120000, 150000, 200000, 250000, 300000], 50000, 55000, legacy=True),
    "eth": Asset("eth", "Ethereum", "2015-07-30", "ETH-USD", "ethusd",
                 [800, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2200, 2500, 3000, 3500, 4000, 5000,
                  6000, 8000, 10000], 1500, 1600,
                 notes="Price history from 2016. The power-law trend uses Ethereum's launch (Jul 2015)."),
    "sol": Asset("sol", "Solana", "2020-03-16", "SOL-USD", "solusd",
                 [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500], 50, 55, sma_min=52,
                 notes="Short history (Coinbase listing, mid-2021): about one full cycle, so the backtest is thin and "
                       "long-range forecasts are much less reliable than for BTC or ETH."),
}
