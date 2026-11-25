"""Who has messaged the bot recently? Lists chat ids so friends can be added.

Telegram keeps a bot's unread messages for 24 hours only, so ask the friend
to send the bot a fresh message just before running this.
Run: Actions -> "Telegram: find chat ids" -> Run workflow.
"""
import json
import os
import urllib.parse
import urllib.request

from .alerts import chats


def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getUpdates?limit=100", timeout=20) as r:
        ups = json.load(r).get("result", [])
    have = set(chats())
    seen = {}
    for u in ups:
        m = u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}
        c = m.get("chat") or {}
        if c.get("id") is not None:
            seen[str(c["id"])] = {"type": c.get("type"), "name": " ".join(x for x in (c.get("first_name"), c.get("last_name")) if x)
                                  or c.get("title", ""), "username": c.get("username", ""), "text": (m.get("text") or "")[:40]}
    lines = ["| Chat id | Name | Username | Type | Already gets alerts |", "|---|---|---|---|---|"]
    for cid, x in seen.items():
        lines.append(f"| `{cid}` | {x['name']} | @{x['username']} | {x['type']} | {'yes' if cid in have else 'no'} |")
        if cid not in have and x["type"] == "private":
            data = urllib.parse.urlencode({"chat_id": cid, "text": f"Hi {x['name'] or 'there'} 👋 Got your message. "
                                           "You'll start receiving the BTC/ETH alerts once you're added."}).encode()
            try:
                urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20)
            except Exception as e:  # noqa: BLE001
                print("reply failed", e)
    out = "\n".join(lines) if seen else "Nobody has messaged the bot in the last 24 hours. Ask your friend to send it a new message, then run again."
    print(out)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write("## People who messaged the bot (last 24h)\n\n" + out + "\n")


if __name__ == "__main__":
    main()
