"""Telegram alerts for the buying plan in plan.json. Runs after the daily pipeline.

Sends a message when:
  - the daily close crosses a buy level (down) or a take-profit level (up)
  - a week closes at/below a reserve trigger or the exit level
  - the chance of a weekly close at/below a watched level by the plan date moves by >= odds_jump_points
  - and a short digest once a week.
Needs repo secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID; without them it prints and does nothing."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd
from .paths import ROOT, DATA, SITE_DATA, DAILY_CSV

STATE = DATA / "alert_state.json"
usd = lambda v: f"${v:,.0f}"


def send(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[alert not sent: no Telegram secrets]\n" + text)
        return False
    chat = chat.strip().strip('"').strip("'")
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "disable_web_page_preview": "true"}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token.strip()}/sendMessage", data=data, timeout=20) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        try:
            reason = json.loads(e.read().decode()).get("description", "")
        except Exception:  # noqa: BLE001
            reason = ""
        print(f"Telegram send failed: HTTP {e.code} {reason}")
        if "chat not found" in reason.lower():
            print("-> TELEGRAM_CHAT_ID is wrong, or you have not sent your bot a message yet. Use the number after "
                  '"chat":{"id": on the getUpdates page (not the "update_id" or "message_id").')
        elif e.code == 401:
            print("-> TELEGRAM_BOT_TOKEN is wrong. Copy it again from @BotFather.")
        return False
    except Exception as e:  # noqa: BLE001
        print("Telegram send failed:", e)
        return False


def _load(name):
    return json.load(open(SITE_DATA / f"{name}.json"))


def odds(risk, week, level, field="p_touch"):
    r = next((x for x in risk if x["week_ending"] == week and x["level"] == level), None)
    return None if r is None else r[field]


def plan_week(forecast, day):
    """The forecast week that contains the plan date (weeks end Saturday)."""
    return next((f["week_ending"] for f in forecast if f["week_ending"] >= day), forecast[-1]["week_ending"])


def main():
    plan = json.load(open(ROOT / "plan.json"))
    if os.environ.get("ALERT_TEST", "").lower() == "true":
        ok = send("✅ Test from your BTC forecast: Telegram alerts are working.\n" + plan.get("page_url", ""))
        print("test alert sent" if ok else "test alert NOT sent")
    daily = pd.read_csv(DAILY_CSV)
    meta, forecast, risk = _load("meta"), _load("forecast"), _load("risk")
    state = json.load(open(STATE)) if STATE.exists() else {}
    today, close = daily["date"].iloc[-1], float(daily["close"].iloc[-1])
    prev = state.get("last_close")
    pw = plan_week(forecast, plan["plan_date"])
    msgs = []

    # 1. daily close crossing buy levels (down) and take-profit levels (up)
    if prev is not None and state.get("last_date") != today:
        for L in plan["buy_ladder"]:
            if prev > L >= close:
                msgs.append(f"🟢 BTC closed at {usd(close)} on {today}, below your buy level {usd(L)}.")
        for L in plan["take_profit"]:
            if prev < L <= close:
                msgs.append(f"💰 BTC closed at {usd(close)} on {today}, above your take-profit level {usd(L)}.")

    # 2. weekly-close triggers (only once the week has closed)
    hist = _load("history")
    done = [h for h in hist if h.get("complete")]
    if done:
        wk, wclose = done[-1]["week_ending"], done[-1]["close"]
        if state.get("last_week_checked") != wk:
            for L in plan["reserve_weekly_close"]:
                if wclose <= L:
                    msgs.append(f"🟠 Reserve trigger: the week ending {wk} closed at {usd(wclose)}, at or below {usd(L)}.")
            if wclose <= plan["exit_weekly_close"]:
                msgs.append(f"🔴 Exit rule: the week ending {wk} closed at {usd(wclose)}, below {usd(plan['exit_weekly_close'])}. "
                            "The cycle model's assumptions are breaking: stop buying and reassess.")
            state["last_week_checked"] = wk

    # 3. odds jumps for the plan date
    old_odds = state.get("odds", {})
    new_odds = {}
    for L in plan["odds_watch_levels"]:
        p = odds(risk, pw, L)
        if p is None:
            continue
        new_odds[str(L)] = p
        o = old_odds.get(str(L))
        if o is not None and abs(p - o) >= plan["odds_jump_points"]:
            arrow = "up" if p > o else "down"
            msgs.append(f"⚠️ Chance of a weekly close ≤ {usd(L)} by {pw} moved {arrow}: {o:.0f}% → {p:.0f}%.")
    state["odds"] = new_odds

    # 4. weekly digest
    now = datetime.now(timezone.utc)
    if now.strftime("%A") == plan.get("weekly_digest_day", "Sunday") and state.get("last_digest") != now.date().isoformat():
        f = next(x for x in forecast if x["week_ending"] == pw)
        nxt = forecast[0]
        lines = [f"📊 Weekly BTC summary ({today})",
                 f"Close: {usd(close)}",
                 f"Next week ({nxt['week_ending']}): {usd(nxt['median'])}, 80% range {usd(nxt['p10'])}–{usd(nxt['p90'])}",
                 f"By {pw}: {usd(f['median'])}, 80% range {usd(f['p10'])}–{usd(f['p90'])}"]
        for L in plan["odds_watch_levels"]:
            if str(L) in new_odds:
                lines.append(f"Chance of a weekly close ≤ {usd(L)} by then: {new_odds[str(L)]:.0f}%")
        lines.append(f"Live forecasts scored so far: {meta.get('live_scored', 0)}")
        lines.append(plan.get("page_url", ""))
        msgs.append("\n".join(lines))
        state["last_digest"] = now.date().isoformat()

    # 5. model health per coin: alert when the status changes (and on the first non-OK reading)
    from .assets import ASSETS
    hstate = state.get("health", {})
    hline = []
    for key, a in ASSETS.items():
        p = a.site_dir / "health.json"
        if not p.exists():
            continue
        h = json.load(open(p))
        st, old = h.get("status", "N/A"), hstate.get(key)
        hline.append(f"{a.name}: {st}")
        if st != old and not (old is None and st in ("OK", "N/A")):
            worst = [r for r in h.get("rows", []) if r["status"] != "OK"]
            detail = "; ".join(f"{r['h']}-week forecasts: {r['cover_recent']:.0f}% inside the 80% range (normal ~80%), "
                               f"typical miss {r['err_recent']:.1f}% vs {r['err_all']:.1f}% usual" for r in worst)
            icon = {"OK": "✅", "WATCH": "🟡", "WARNING": "🔴"}.get(st, "ℹ️")
            msgs.append(f"{icon} Model health for {a.name}: {old or 'new'} → {st}." + (f" {detail}." if detail else ""))
        hstate[key] = st
    state["health"] = hstate
    if hline and msgs and msgs[-1].startswith("📊"):
        msgs[-1] += "\nModel health: " + ", ".join(hline)

    state.update(last_close=close, last_date=today)
    json.dump(state, open(STATE, "w"), indent=1)
    if msgs:
        send("\n\n".join(msgs) + "\n\nNot financial advice.")
    print(f"alerts: {len(msgs)} message(s)")


if __name__ == "__main__":
    main()
