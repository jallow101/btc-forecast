from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SITE_DATA = ROOT / "docs" / "data"
DAILY_CSV = DATA / "btc-usd-daily.csv"
BACKTEST_CSV = DATA / "backtest_preds.csv"
FORECAST_LOG = DATA / "forecast_log.csv"
DATA.mkdir(exist_ok=True)
SITE_DATA.mkdir(parents=True, exist_ok=True)
