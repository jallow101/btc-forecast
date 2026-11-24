"""Telegram price alerts for the re-entry zones in data/zones.json.

Checks every closed 15m candle since the last run (so a delayed run can't
miss a wick). Each zone fires once, then re-arms after price moves 2% back
away from it. State in out/trades/zone_state.json (published with the log).
"""
import json

import pandas as pd

from .paths import DATA, ROOT
from .scalp import fmtp

STATE = ROOT / "out" / "trades" / "zone_state.json"
REARM = 0.02


def candles(coin):
    df = pd.read_csv(DATA / "liq" / f"{coin}_15m.csv")
    return df.iloc[:-1]                        # last row is still forming


def run(derivs=None):
    from .alerts import send
    cfg = json.loads((DATA / "zones.json").read_text())["zones"]
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    fired = []
    for z in cfg:
        s = st.setdefault(z["id"], {})
        try:
            df = candles(z["coin"])
        except FileNotFoundError:
            continue
        last = s.get("last_ts")
        new = df[df.ts > last] if last else df.tail(1)
        if new.empty:
            continue
        lvl, down = z["price"], z["dir"] == "down"
        if "armed" not in s:                    # first sight: arm only if price is on the other side, no alert
            px = float(new.close.iat[-1])
            s["armed"] = px > lvl if down else px < lvl
            s["last_ts"] = int(new.ts.iat[-1])
            continue
        for r in new.itertuples():
            if s["armed"] and ((down and r.low <= lvl) or (not down and r.high >= lvl)):
                s.update(armed=False, fired_at=int(r.ts), fired_px=float(r.close))
                fired.append((z, r))
            elif not s["armed"] and ((down and r.close >= lvl * (1 + REARM)) or
                                     (not down and r.close <= lvl * (1 - REARM))):
                s["armed"] = True
        s["last_ts"] = int(new.ts.iat[-1])
    for z, r in fired:
        t = pd.to_datetime(r.ts, unit="s", utc=True).strftime("%d %b %H:%M")
        msg = (f"{'🔻' if z['dir'] == 'down' else '🔺'} {z['title']}\n"
               f"{z['coin']} touched {fmtp(z['price'])} ({t} UTC candle, closed {fmtp(r.close)})\n"
               f"Plan: {z['plan']}")
        d = (derivs or {}).get(z["coin"])
        if d and d.get("read"):
            msg += "\nLeverage now: " + " ".join(d["read"][:2])
        msg += "\nCheck the price on Bybit before placing orders."
        print(msg)
        send(msg)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))
    return fired


if __name__ == "__main__":
    run()
