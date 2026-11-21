"""Does the liquidity map carry information? Two honest tests.

Test A - direction ("which cluster gets tapped first")
  At regular points, take the strongest cluster above and below price. Under
  a driftless random walk the chance of hitting the upper one first depends
  only on distance:  p_up = ln(P/down) / (ln(up/P) + ln(P/down)).
  So "price taps clusters" is trivially true; the real question is whether the
  HEAVIER side gets hit first more often than distance alone predicts.
  Excess = actual up-first rate - distance-only expectation, measured when the
  map is heavier above vs heavier below. The difference between those two
  excesses is the map's edge (it also cancels any trend in the sample).

Test B - sweep & reclaim ("liquidity grab, then reversal")
  Event: a bar wicks through a large amount of resting liquidity (top 10% of
  swept weight) and closes back beyond the nearest swept level. Forward
  return in the reversal direction is compared with a control: bars that
  broke the 20-bar extreme and closed back inside, but swept little liquidity.

Test C - "once direction is confirmed it taps the clusters"
  Trend = close above a rising 50-bar average (up) or below a falling one
  (down). Within each trend, compare the heavy-side excess when the heavy
  cluster is in the trend direction vs against it. Their gap isolates the
  liquidity effect from plain momentum.

Nothing was tuned on this data: all settings are the defaults in liquidity.py.
Block bootstrap (blocks = one horizon) gives 90% intervals, because
overlapping windows are not independent.
"""
import json

import numpy as np
import pandas as pd

from . import liquidity as L
from .paths import ROOT

EVAL_EVERY = {"15m": 4, "1h": 4, "4h": 2}
WARMUP = 300
FWD = {"15m": [4, 16, 48], "1h": [4, 12, 24], "4h": [3, 6, 12]}
RNG = np.random.default_rng(7)


def strongest(eng, i, price, cfg):
    px, w, above, _ = eng.weights(i)
    if len(px) == 0:
        return None
    d = px / price - 1
    out = {}
    for side, m in (("up", (d >= cfg["min"]) & (d <= cfg["range"])),
                    ("dn", (d <= -cfg["min"]) & (d >= -cfg["range"]))):
        if not m.any():
            return None
        step = price * cfg["bin"]
        b = np.floor((px[m] - price) / step).astype(int)
        s = pd.Series(w[m]).groupby(b).sum()
        k = s.idxmax()
        out[side] = (price + (k + 0.5) * step, float(s.max()))
    return out


def collect(coin, tf):
    cfg = L.TF_CFG[tf]
    df = L.load(coin, tf)
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n, H, E = len(df), cfg["horizon"], EVAL_EVERY[tf]
    dir_rows, sweeps = [], []

    def cb(i, eng, swept):
        sb, sa, pb, pa = swept
        sweeps.append((i, sb, sa, pb, pa))
        if i < WARMUP or i + H >= n or (i - WARMUP) % E:
            return
        s = strongest(eng, i, c[i], cfg)
        if not s:
            return
        up, wu = s["up"]
        dn, wd = s["dn"]
        hit = None
        for j in range(i + 1, i + H + 1):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                hit = None
                break
            if hu or hd:
                hit = 1 if hu else 0
                break
        if hit is None:
            return
        a, b = np.log(c[i] / dn), np.log(up / c[i])
        dir_rows.append({"i": i, "up_first": hit, "p_dist": a / (a + b), "imb": np.log(wu / wd),
                         "d_up": b, "d_dn": a})

    L.run_engine(df, cb)
    d = pd.DataFrame(dir_rows)
    sw = pd.DataFrame(sweeps, columns=["i", "sb", "sa", "pb", "pa"]).set_index("i")
    return df, d, sw


def boot(vals, block, stat=np.mean, reps=2000):
    vals = np.asarray(vals, float)
    n = len(vals)
    if n < 10:
        return (np.nan, np.nan)
    nb = max(1, n // block)
    starts = RNG.integers(0, max(1, n - block + 1), size=(reps, nb))
    res = [stat(np.concatenate([vals[s:s + block] for s in row])) for row in starts]
    return tuple(np.percentile(res, [5, 95]))


def test_direction(d, tf):
    if d.empty:
        return {}
    blk = max(2, L.TF_CFG[tf]["horizon"] // EVAL_EVERY[tf])
    d = d.copy()
    d["excess"] = d.up_first - d.p_dist
    heavy_up, heavy_dn = d[d.imb > np.log(1.5)], d[d.imb < -np.log(1.5)]
    # edge: excess when heavier above minus excess when heavier below
    sign = np.where(d.imb > 0, 1, -1)
    edge_series = d.excess * sign
    # simple rule: predict the side with more weight per unit distance
    score = d.imb - np.log(d.d_up / d.d_dn)
    rule_hit = ((score > 0).astype(int) == d.up_first).mean()
    dist_hit = ((d.p_dist > 0.5).astype(int) == d.up_first).mean()
    return {
        "n": int(len(d)),
        "up_first_rate": round(d.up_first.mean(), 3),
        "distance_expectation": round(d.p_dist.mean(), 3),
        "heavy_above": {"n": int(len(heavy_up)), "actual": round(heavy_up.up_first.mean(), 3),
                        "expected": round(heavy_up.p_dist.mean(), 3),
                        "excess": round(heavy_up.excess.mean(), 3)},
        "heavy_below": {"n": int(len(heavy_dn)), "actual": round(heavy_dn.up_first.mean(), 3),
                        "expected": round(heavy_dn.p_dist.mean(), 3),
                        "excess": round(heavy_dn.excess.mean(), 3)},
        "edge": round(float(edge_series.mean()), 3),
        "edge_ci90": [round(x, 3) for x in boot(edge_series.to_numpy(), blk)],
        "rule_accuracy": round(float(rule_hit), 3),
        "nearest_side_accuracy": round(float(dist_hit), 3),
    }


def add_trend(d, df):
    c = df.close
    ma = c.rolling(50).mean()
    sl = ma - ma.shift(10)
    ci, mi, si = c[d.i].to_numpy(), ma[d.i].to_numpy(), sl[d.i].to_numpy()
    d = d.copy()
    d["trend"] = np.where((ci > mi) & (si > 0), 1, np.where((ci < mi) & (si < 0), -1, 0))
    d["heavy"] = np.where(d.imb > np.log(1.5), 1, np.where(d.imb < -np.log(1.5), -1, 0))
    d["hx"] = (d.up_first - d.p_dist) * d.heavy        # excess toward the heavy side
    return d


def test_trend(d, tf):
    blk = max(2, L.TF_CFG[tf]["horizon"] // EVAL_EVERY[tf])
    m = (d.trend != 0) & (d.heavy != 0)
    al, co = d[m & (d.trend == d.heavy)], d[m & (d.trend != d.heavy)]
    out = {"with_trend": {"n": int(len(al)), "excess": round(float(al.hx.mean()), 3),
                          "ci90": [round(x, 3) for x in boot(al.hx.to_numpy(), blk)]},
           "against_trend": {"n": int(len(co)), "excess": round(float(co.hx.mean()), 3),
                             "ci90": [round(x, 3) for x in boot(co.hx.to_numpy(), blk)]}}
    gaps = []
    for t in (1, -1):
        s = d[d.trend == t]
        a = s[s.heavy == t].hx.mean()        # heavy toward trend, excess toward heavy
        b = (-s[s.heavy == -t].hx).mean()    # heavy against trend, excess toward trend
        gaps.append(a - b)
        out["uptrend_gap" if t == 1 else "downtrend_gap"] = round(float(a - b), 3)
    return out


def test_sweeps(df, sw, tf):
    c = df.close.to_numpy(float)
    h = df.high.to_numpy(float)
    l = df.low.to_numpy(float)
    n = len(c)
    fw = FWD[tf]
    lo20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    hi20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    sw = sw.reindex(range(n))
    thr_b = sw.sb[sw.sb > 0].quantile(0.9)
    thr_a = sw.sa[sw.sa > 0].quantile(0.9)
    small_b = sw.sb[sw.sb > 0].quantile(0.5)
    small_a = sw.sa[sw.sa > 0].quantile(0.5)
    rows = []
    last = {"ev": -99, "ct": -99}
    for i in range(WARMUP, n - max(fw)):
        sb, sa, pb, pa = sw.sb[i], sw.sa[i], sw.pb[i], sw.pa[i]
        # long setup: grab below, close back above the first swept level
        if sb >= thr_b and pb == pb and c[i] > pb:
            grp, sgn = "event", 1
        elif sa >= thr_a and pa == pa and c[i] < pa:
            grp, sgn = "event", -1
        elif l[i] < lo20[i] and c[i] > lo20[i] and sb <= small_b:
            grp, sgn = "control", 1
        elif h[i] > hi20[i] and c[i] < hi20[i] and sa <= small_a:
            grp, sgn = "control", -1
        else:
            continue
        key = "ev" if grp == "event" else "ct"
        if i - last[key] < 3:          # one signal per cluster of bars
            continue
        last[key] = i
        r = {"i": i, "grp": grp, "side": "long" if sgn > 0 else "short"}
        for k in fw:
            r[f"r{k}"] = sgn * np.log(c[i + k] / c[i]) * 100
        rows.append(r)
    t = pd.DataFrame(rows)
    out = {"thresholds": {"below": round(float(thr_b), 2), "above": round(float(thr_a), 2)}}
    for g in ("event", "control"):
        s = t[t.grp == g]
        out[g] = {"n": int(len(s))}
        for k in fw:
            v = s[f"r{k}"]
            out[g][f"{k}"] = {"mean_pct": round(v.mean(), 3), "win": round((v > 0).mean(), 3),
                              "ci90": [round(x, 3) for x in boot(v.to_numpy(), 1)]}
    return out, t


def run():
    res = {}
    pooled_dir, pooled_sw = {tf: [] for tf in L.TF_CFG}, {tf: [] for tf in L.TF_CFG}
    for coin in L.COINS:
        for tf in L.TF_CFG:
            df, d, sw = collect(coin, tf)
            a = test_direction(d, tf)
            b, t = test_sweeps(df, sw, tf)
            res[f"{coin}_{tf}"] = {"period": [str(df.t.iat[WARMUP].date()), str(df.t.iat[-1].date())],
                                   "direction": a, "sweeps": b}
            d = add_trend(d, df) if not d.empty else d
            d["coin"] = coin
            t["coin"] = coin
            pooled_dir[tf].append(d)
            pooled_sw[tf].append(t)
            print(coin, tf, json.dumps(a), json.dumps({g: b[g] for g in ("event", "control")}))
    pooled = {}
    for tf in L.TF_CFG:
        d = pd.concat(pooled_dir[tf], ignore_index=True)
        t = pd.concat(pooled_sw[tf], ignore_index=True)
        p = {"direction": test_direction(d, tf), "trend": test_trend(d, tf), "sweeps": {},
             "period": [min(r["period"][0] for k, r in res.items() if k.endswith("_" + tf)),
                        max(r["period"][1] for k, r in res.items() if k.endswith("_" + tf))]}
        for g in ("event", "control"):
            s = t[t.grp == g]
            p["sweeps"][g] = {"n": int(len(s))}
            for k in FWD[tf]:
                v = s[f"r{k}"]
                p["sweeps"][g][str(k)] = {"mean_pct": round(v.mean(), 3), "win": round((v > 0).mean(), 3),
                                          "ci90": [round(x, 3) for x in boot(v.to_numpy(), 1)]}
        pooled[tf] = p
    out = {"per_series": res, "pooled": pooled}
    (rep_dir := ROOT / "reports").mkdir(exist_ok=True)
    (rep_dir / "liquidity_backtest.md").write_text(markdown(out))
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "liquidity_backtest.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(pooled, indent=1, default=float))
    return out


def pp(x):
    return f"{x * 100:+.1f} pts"


def markdown(out):
    L_ = ["# Liquidity map backtest", "",
          "Candle-based liquidity map (estimated liquidation levels at 10/25/50/100x + swing stops),",
          "Hyperliquid perp candles for BTC, ETH, SOL, WLD. No settings were tuned on this data.", ""]
    for tf, p in out["pooled"].items():
        d, t, s = p["direction"], p["trend"], p["sweeps"]
        k = list(s["event"])[1]
        L_ += [f"## {tf}  ({p['period'][0]} to {p['period'][1]}, {d['n']} test points)", "",
               "| Test | Result | 90% range |", "|---|---|---|",
               f"| A. Heavier side hit first, beyond distance | {pp(d['edge'])} | {pp(d['edge_ci90'][0])} to {pp(d['edge_ci90'][1])} |",
               f"| C. Heavy cluster WITH the trend | {pp(t['with_trend']['excess'])} | {pp(t['with_trend']['ci90'][0])} to {pp(t['with_trend']['ci90'][1])} |",
               f"| C. Heavy cluster AGAINST the trend | {pp(t['against_trend']['excess'])} | {pp(t['against_trend']['ci90'][0])} to {pp(t['against_trend']['ci90'][1])} |",
               f"| B. Sweep & reclaim, {k} bars later (n={s['event']['n']}) | {s['event'][k]['mean_pct']:+.2f}% , win {s['event'][k]['win']:.0%} | {s['event'][k]['ci90'][0]:+.2f}% to {s['event'][k]['ci90'][1]:+.2f}% |",
               f"| B. Control: plain wick & reclaim (n={s['control']['n']}) | {s['control'][k]['mean_pct']:+.2f}% , win {s['control'][k]['win']:.0%} | {s['control'][k]['ci90'][0]:+.2f}% to {s['control'][k]['ci90'][1]:+.2f}% |",
               f"| Nearest-side rule accuracy / map rule accuracy | {d['nearest_side_accuracy']:.0%} / {d['rule_accuracy']:.0%} | |",
               ""]
    return "\n".join(L_)


if __name__ == "__main__":
    run()
