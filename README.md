# btc-forecast

Daily-retrained, backtested weekly price forecasts for BTC, ETH and SOL (up to 4 years ahead), with live tracking of every forecast against what actually happened.

**Live page:** https://jallow101.github.io/btc-forecast/ (published by the workflow; Settings → Pages → Source: GitHub Actions). **Runs:** every day at 01:40 UTC via GitHub Actions (`.github/workflows/daily.yml`), or by hand from the Actions tab.

## What each run does
1. `btcf/fetch.py` — adds new daily BTC/USD candles from the Bitstamp public API (fallback: github.com/rcaljouw/btc-price-history).
2. `btcf/features.py` — weekly closes (weeks end Saturday) + signals: distance from 200-week MA, drawdown from ATH, weeks since ATH, halving-cycle phase, momentum, volatility.
3. `btcf/backtest.py` — walk-forward backtest 2017→now that mirrors the live model: each week's forecast uses only data known that week (history refit every 4 weeks; new weeks are fit one at a time). Results are stored in `data/backtest_preds.csv` and only new weeks are added each run.
4. `btcf/forecast.py` — ensemble of 6-yr return climatology + gradient-boosted quantile regression + nearest-neighbour analogs, 80% band calibrated on the backtest, 20,000 simulated paths.
5. `btcf/track.py` — appends the forecast to `data/forecast_log.csv` and scores every past forecast whose week has closed.
6. `btcf/site.py` — writes `docs/data/*.json` for the page (not committed; deployed with the page).

Only `data/btc-usd-daily.csv`, `data/forecast_log.csv` and `data/backtest_preds.csv` are committed each day, so the forecast log is a permanent, versioned record of every forecast.

Run locally: `pip install -r requirements.txt && python -m btcf.pipeline`

Not financial advice.

## Coins
`btcf/assets.py` lists the coins (BTC, ETH, SOL): data source, price levels, and where each coin's files live
(BTC in `data/`, others in `data/<coin>/`). The page switches coin with `?coin=eth` / `?coin=sol`.

## Model health
`btcf/health.py` compares the last 13–26 weeks of forecasts with each coin's long-run record (share of outcomes
inside the 80% range, typical miss). Status OK / WATCH / WARNING is shown on the page, and a Telegram alert is
sent when it changes.
