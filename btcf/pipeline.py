"""Daily job, for every coin in assets.ASSETS: fetch prices -> rebuild features -> extend the
walk-forward backtest -> forecast -> log the forecast -> score past forecasts -> health check
-> write the web page's data. One coin failing does not stop the others."""
import json
import sys
import time
import traceback

from . import fetch, features, backtest, forecast, track, site, health
from .assets import ASSETS
from .paths import SITE_DATA


def run_asset(a):
    t0 = time.time()
    daily = fetch.update_asset(a)
    w = features.build(a)
    P = backtest.run(w, store=a.backtest_csv, genesis=a.genesis)
    fc = forecast.run(w, P, levels=a.levels, genesis=a.genesis)
    log = track.log_run(fc, run_date=str(daily["date"].iloc[-1]), path=a.forecast_log)
    scored = track.evaluate(log, w)
    site.write(w, P, fc, scored, asset=a)
    hc = health.check(w, P)
    json.dump(hc, open(a.site_dir / "health.json", "w"), indent=1)
    i = fc["info"]
    print(f"{a.key}: as of {i['last_day']} price {i['price']:,}  health {hc['status']}  "
          f"backtest {len(P)} rows  live scored {len(scored)}  ({time.time() - t0:.0f}s)")
    return {"key": a.key, "name": a.name, "price": i["price"], "last_day": i["last_day"], "health": hc["status"]}


def main():
    done, failed = [], []
    for key, a in ASSETS.items():
        try:
            done.append(run_asset(a))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append(key)
    json.dump({"assets": done, "failed": failed}, open(SITE_DATA / "assets.json", "w"), indent=1)
    try:                            # daily highs / lows, 1-7 days ahead
        from . import daily_range
        daily_range.run_live()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    if "btc" in failed:
        sys.exit(1)                 # BTC is the core: fail the run so it shows red on GitHub


if __name__ == "__main__":
    main()
