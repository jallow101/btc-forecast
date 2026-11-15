# btc-forecast

Daily-retrained, backtested Bitcoin weekly price forecast (up to 4 years ahead), with live tracking of every forecast against what actually happened.

**Live page:** https://jallow101.github.io/btc-forecast/ (published by the workflow; Settings → Pages → Source: GitHub Actions). **Runs:** every day at 01:40 UTC via GitHub Actions (`.github/workflows/daily.yml`), or by hand from the Actions tab.

## What each run does
1. `btcf/fetch.py` — adds new daily BTC/USD candles from the Bitstamp public API (fallback: github.com/rcaljouw/btc-price-history).
2. `btcf/features.py` — weekly closes (weeks end Saturday) + signals: distance from 200-week MA, drawdown from ATH, weeks since ATH, halving-cycle phase, momentum, volatility.
3. `btcf/backtest.py` — walk-forward backtest 2017→now; every model is retrained per test year using only data known before it.
4. `btcf/forecast.py` — ensemble of 6-yr return climatology + gradient-boosted quantile regression + nearest-neighbour analogs, 80% band calibrated on the backtest, 20,000 simulated paths.
5. `btcf/track.py` — appends the forecast to `data/forecast_log.csv` and scores every past forecast whose week has closed.
6. `btcf/site.py` — writes `docs/data/*.json` for the page (not committed; deployed with the page).

Only `data/btc-usd-daily.csv` and `data/forecast_log.csv` are committed each day, so the forecast log is a permanent, versioned record of every forecast.

Run locally: `pip install -r requirements.txt && python -m btcf.pipeline`

Not financial advice.
