"""Which free open-interest / liquidation endpoints answer from GitHub's runners?"""
import json
import urllib.request

UA = {"User-Agent": "btc-forecast/1.0", "Content-Type": "application/json"}
TESTS = {
    "okx_oi_hist": "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history?instId=BTC-USDT-SWAP&period=1H&limit=100",
    "okx_liq": "https://www.okx.com/api/v5/public/liquidation-orders?instType=SWAP&uly=BTC-USDT&state=filled&limit=5",
    "bybit_oi": "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1h&limit=5",
    "binance_oi": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=5",
}
for k, u in TESTS.items():
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=20) as r:
            body = r.read()[:300]
        print(k, "OK", body)
    except Exception as e:  # noqa: BLE001
        print(k, "FAIL", e)
try:
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
                                 data=json.dumps({"type": "metaAndAssetCtxs"}).encode(), headers=UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        js = json.load(r)
    names = [a["name"] for a in js[0]["universe"]]
    for c in ["BTC", "ETH", "SOL", "WLD"]:
        if c in names:
            print("hl", c, js[1][names.index(c)].get("openInterest"))
        else:
            print("hl", c, "not listed")
except Exception as e:  # noqa: BLE001
    print("hyperliquid ctx FAIL", e)
