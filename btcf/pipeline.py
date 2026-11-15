"""Daily job: fetch prices -> rebuild features -> walk-forward backtest (retrains every model)
-> forecast -> log the forecast -> score past forecasts -> write the web page's data."""
import time
from . import fetch, features, backtest, forecast, track, site


def main():
    t0 = time.time()
    daily = fetch.update()
    w = features.build()
    print("weeks:", len(w), "latest:", w.index[-1].date(), "complete:", bool(w.complete.iloc[-1]))
    P = backtest.run(w)
    print(f"backtest rows: {len(P)} ({time.time() - t0:.0f}s)")
    fc = forecast.run(w, P)
    log = track.log_run(fc, run_date=str(daily["date"].iloc[-1]))
    scored = track.evaluate(log, w)
    site.write(w, P, fc, scored)
    i = fc["info"]
    print(f"as of {i['last_day']}: price {i['price']:,}  median {fc['weekly']['median'].iloc[-1]:,.0f} by {i['end_week']}  "
          f"scored live forecasts: {len(scored)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
