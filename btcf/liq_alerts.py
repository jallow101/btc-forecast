"""Telegram alerts from the liquidity map (runs hourly).

* approaching: price within APPROACH of the biggest cluster above / below (1h and 4h)
* swept: price ran through the biggest cluster that was there on the previous run
Each message says whether the heavy side agrees with the trend, the only case
where the backtest found an edge. State lives next to the site files.
"""
import json
import sys
import time
from pathlib import Path

from .alerts import send

COINS = ["BTC", "ETH", "SOL", "WLD"]
TFS = ["1h", "4h"]
APPROACH = 0.006
COOLDOWN_H = 12


def fmt(p):
    return f"${p:,.0f}" if p >= 1000 else f"${p:,.2f}" if p >= 10 else f"${p:.4f}"


def main(site: Path):
    summary = json.loads((site / "summary.json").read_text())
    sp = site / "alert_state.json"
    state = json.loads(sp.read_text()) if sp.exists() else {"sent": {}, "prev": {}}
    now = time.time()
    lines = []
    for coin in COINS:
        for tf in TFS:
            s = summary.get(f"{coin}_{tf}")
            if not s:
                continue
            price, trend, mj = s["price"], s["trend"], s["majors"]
            up = mj["above"][0] if mj["above"] else None
            dn = mj["below"][0] if mj["below"] else None
            heavy = None
            if up and dn:
                heavy = "above" if up["weight"] > 1.5 * dn["weight"] else "below" if dn["weight"] > 1.5 * up["weight"] else None
            prev = state["prev"].get(f"{coin}_{tf}")
            # swept since last run
            if prev:
                for side, lvl in (("above", prev.get("above")), ("below", prev.get("below"))):
                    if lvl and ((side == "above" and price >= lvl * 0.999) or (side == "below" and price <= lvl * 1.001)):
                        lines.append(f"🧹 {coin} {tf}: swept the {side} cluster at {fmt(lvl)} (now {fmt(price)})")
            for side, c in (("above", up), ("below", dn)):
                if not c:
                    continue
                dist = abs(c["price"] / price - 1)
                key = f"{coin}_{tf}_{side}_{c['price']:.4g}"
                if dist <= APPROACH and now - state["sent"].get(key, 0) > COOLDOWN_H * 3600:
                    note = ""
                    if heavy == side and trend != "none":
                        note = " — heavy side WITH the trend (backtest edge)" if (side == "above") == (trend == "up") else " — heavy side against the trend (weak)"
                    lines.append(f"🎯 {coin} {tf}: {dist * 100:.2f}% from the biggest {side} cluster {fmt(c['price'])}"
                                 f" (strength {c['weight']:.1f}, {trend}trend){note}")
                    state["sent"][key] = now
            state["prev"][f"{coin}_{tf}"] = {"above": up["price"] if up else None, "below": dn["price"] if dn else None}
    state["sent"] = {k: v for k, v in state["sent"].items() if now - v < 7 * 86400}
    sp.write_text(json.dumps(state, indent=1))
    if lines:
        msg = "Liquidity map\n" + "\n".join(lines) + "\nhttps://jallow101.github.io/btc-forecast/liquidity.html"
        print(msg)
        send(msg)
    else:
        print("no liquidity alerts")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/data/liquidity"))
