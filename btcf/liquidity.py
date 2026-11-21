"""Liquidity clusters from candles alone (free, and identical to the Pine script).

Two kinds of resting liquidity are tracked bar by bar:

1. Estimated liquidation levels. A candle with unusually high volume
   (> VOL_MULT x its 50-bar average) is treated as a burst of new leveraged
   positions opened at its close. For each leverage tier the liquidation price
   of a long (below) and a short (above) is added:
       long  liq = close * (1 - 1/L + MMR)
       short liq = close * (1 + 1/L - MMR)
   weighted by relative volume x tier share.
2. Swing stops. Confirmed pivot highs / lows (5 bars each side): stops of
   shorts sit above swing highs, stops of longs below swing lows.

A level is removed once price trades through it (those positions are gone),
and its weight halves every HALF_LIFE bars (positions get closed anyway).
Active levels are summed into price bins -> the heatmap.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .paths import DATA, ROOT

LEVERAGE = {10: 0.30, 25: 0.30, 50: 0.25, 100: 0.15}
MMR = 0.005
VOL_LEN = 50
VOL_MULT = 1.5
PIVOT = 5
SWING_W = 1.0
HALF_LIFE = 250
MAX_AGE = 1000

# per timeframe: bin width, search range, minimum distance (fractions of price)
TF_CFG = {
    "15m": {"bin": 0.0025, "range": 0.04, "min": 0.003, "horizon": 96},
    "1h": {"bin": 0.0025, "range": 0.08, "min": 0.005, "horizon": 96},
    "4h": {"bin": 0.005, "range": 0.20, "min": 0.01, "horizon": 42},
}
COINS = ["BTC", "ETH", "SOL", "WLD"]


@dataclass
class Engine:
    px: list = field(default_factory=list)      # level price
    w: list = field(default_factory=list)       # base weight
    born: list = field(default_factory=list)    # bar index
    above: list = field(default_factory=list)   # True = above price (short liq / swing high)
    kind: list = field(default_factory=list)    # "liq" or "swing"

    def add(self, price, weight, i, above, kind):
        self.px.append(price)
        self.w.append(weight)
        self.born.append(i)
        self.above.append(above)
        self.kind.append(kind)

    def sweep(self, i, high, low):
        """Remove levels price traded through.

        Returns swept weight below / above and the swept level nearest to the
        prior price on each side (the first one price ran into)."""
        keep = []
        sb = sa = 0.0
        pb = pa = None
        for k in range(len(self.px)):
            age = i - self.born[k]
            if age > MAX_AGE:
                continue
            wk = self.w[k] * 0.5 ** (age / HALF_LIFE)
            if self.above[k] and high >= self.px[k]:
                sa += wk
                pa = self.px[k] if pa is None else min(pa, self.px[k])
                continue
            if not self.above[k] and low <= self.px[k]:
                sb += wk
                pb = self.px[k] if pb is None else max(pb, self.px[k])
                continue
            keep.append(k)
        for name in ("px", "w", "born", "above", "kind"):
            arr = getattr(self, name)
            setattr(self, name, [arr[k] for k in keep])
        return sb, sa, pb, pa

    def weights(self, i):
        if not self.px:
            return np.array([]), np.array([]), np.array([], bool), np.array([])
        age = i - np.array(self.born)
        return (np.array(self.px), np.array(self.w) * 0.5 ** (age / HALF_LIFE),
                np.array(self.above), np.array(self.kind))


def run_engine(df, on_bar=None):
    """Walk the candles. on_bar(i, engine, swept) is called after each bar."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    vavg = pd.Series(v).rolling(VOL_LEN).mean().shift(1).to_numpy()
    eng = Engine()
    for i in range(len(df)):
        swept = eng.sweep(i, h[i], l[i])
        # confirmed pivot at i - PIVOT
        j = i - PIVOT
        if j >= PIVOT:
            win_h = h[j - PIVOT:i + 1]
            win_l = l[j - PIVOT:i + 1]
            rel_j = v[j] / vavg[j] if vavg[j] and not math.isnan(vavg[j]) else 1.0
            sw = SWING_W * max(1.0, rel_j)
            # a pivot level only counts if price has not already traded through it
            if h[j] == win_h.max() and h[j] > max(h[j + 1:i + 1]):
                eng.add(h[j], sw, i, True, "swing")
            if l[j] == win_l.min() and l[j] < min(l[j + 1:i + 1]):
                eng.add(l[j], sw, i, False, "swing")
        if not math.isnan(vavg[i]) and vavg[i] > 0:
            rel = v[i] / vavg[i]
            if rel > VOL_MULT:
                for lev, share in LEVERAGE.items():
                    eng.add(c[i] * (1 - 1 / lev + MMR), rel * share, i, False, "liq")
                    eng.add(c[i] * (1 + 1 / lev - MMR), rel * share, i, True, "liq")
        if on_bar:
            on_bar(i, eng, swept)
    return eng


def bins(eng, i, price, bin_w, rng):
    """Sum active weights into bins of bin_w (fraction of price) within +-rng."""
    px, w, above, kind = eng.weights(i)
    if len(px) == 0:
        return pd.DataFrame(columns=["lo", "hi", "mid", "weight", "liq", "swing", "side"])
    step = price * bin_w
    m = (px > price * (1 - rng)) & (px < price * (1 + rng))
    px, w, above, kind = px[m], w[m], above[m], kind[m]
    idx = np.floor((px - price) / step).astype(int)
    df = pd.DataFrame({"b": idx, "w": w, "liq": np.where(kind == "liq", w, 0.0),
                       "swing": np.where(kind == "swing", w, 0.0)})
    g = df.groupby("b").sum()
    g["lo"] = price + g.index * step
    g["hi"] = g["lo"] + step
    g["mid"] = (g["lo"] + g["hi"]) / 2
    g["side"] = np.where(g["mid"] > price, "above", "below")
    return g.rename(columns={"w": "weight"}).reset_index(drop=True)[
        ["lo", "hi", "mid", "weight", "liq", "swing", "side"]]


def majors(b, price, min_d, top=3):
    """Strongest bins on each side (merge touching bins first)."""
    out = {}
    for side in ("above", "below"):
        s = b[(b.side == side) & ((b.mid / price - 1).abs() >= min_d)].sort_values("mid")
        if s.empty:
            out[side] = []
            continue
        # merge adjacent bins into clusters
        groups, cur = [], [s.iloc[0]]
        for r in s.iloc[1:].itertuples(index=False):
            if abs(r.lo - cur[-1].hi) < 1e-9 * price:
                cur.append(pd.Series(r._asdict()))
            else:
                groups.append(cur)
                cur = [pd.Series(r._asdict())]
        groups.append(cur)
        cl = []
        for g in groups:
            g = pd.DataFrame(g)
            wsum = g.weight.sum()
            cl.append({"lo": float(g.lo.min()), "hi": float(g.hi.max()),
                       "price": float((g.mid * g.weight).sum() / wsum), "weight": float(wsum),
                       "liq_share": float(g.liq.sum() / wsum)})
        cl.sort(key=lambda x: -x["weight"])
        out[side] = cl[:top]
    return out


def load(coin, tf):
    df = pd.read_csv(DATA / "liq" / f"{coin}_{tf}.csv")
    df["t"] = pd.to_datetime(df.ts, unit="s", utc=True)
    return df.reset_index(drop=True)


def snapshot(coin, tf, heat_bars=240, every=2):
    """Current clusters + a compact time x price heatmap of the last heat_bars bars."""
    cfg = TF_CFG[tf]
    df = load(coin, tf)
    n = len(df)
    start = n - heat_bars
    last = float(df.close.iat[-1])
    seen = df.iloc[start:]
    rng = max(cfg["range"] * 1.5, seen.high.max() / last - 1 + 0.05, 1 - seen.low.min() / last + 0.05)
    step = last * cfg["bin"]
    lo_edge = last * (1 - rng)
    nb = int(np.ceil(2 * rng / cfg["bin"]))
    times, mat = [], []

    def cb(i, eng, swept):
        if i >= start and ((i - start) % every == 0 or i == n - 1):
            px, w, _, _ = eng.weights(i)
            row = np.zeros(nb)
            if len(px):
                k = np.floor((px - lo_edge) / step).astype(int)
                m = (k >= 0) & (k < nb)
                np.add.at(row, k[m], w[m])
            times.append(df.t.iat[i].isoformat())
            mat.append([round(x, 2) for x in row])

    eng = run_engine(df, cb)
    b = bins(eng, n - 1, last, cfg["bin"], cfg["range"])
    mj = majors(b, last, cfg["min"])
    for side in mj.values():
        for c in side:
            c["dist_pct"] = round((c["price"] / last - 1) * 100, 2)
    candles = df.iloc[start:][["t", "open", "high", "low", "close"]].copy()
    candles["t"] = candles.t.map(lambda x: x.isoformat())
    # trend context used by the backtest's test C
    ma = df.close.rolling(50).mean()
    slope = ma.iat[-1] - ma.iat[-11]
    trend = "up" if last > ma.iat[-1] and slope > 0 else "down" if last < ma.iat[-1] and slope < 0 else "none"
    return {
        "coin": coin, "tf": tf, "asof": df.t.iat[-1].isoformat(), "price": last, "trend": trend,
        "bin": cfg["bin"], "majors": mj,
        "levels": [[round(r.mid, 6), round(r.weight, 3), round(r.liq, 3), round(r.swing, 3)] for r in b.itertuples()],
        "grid": {"lo": lo_edge, "step": step, "n": nb},
        "heat": {"t": times, "w": mat},
        "candles": candles.round(6).values.tolist(),
    }


def write_site(out=None):
    out = out or (ROOT / "docs" / "data" / "liquidity")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for coin in COINS:
        for tf in TF_CFG:
            try:
                s = snapshot(coin, tf, every=1 if tf == "15m" else 2)
            except FileNotFoundError:
                continue
            (out / f"{coin}_{tf}.json").write_text(json.dumps(s, separators=(",", ":")))
            summary[f"{coin}_{tf}"] = {"price": s["price"], "asof": s["asof"], "trend": s["trend"],
                                       "majors": s["majors"]}
            print(coin, tf, s["price"], s["trend"],
                  {k: [(round(c["price"], 4), round(c["weight"], 1)) for c in v] for k, v in s["majors"].items()})
    bt = ROOT / "reports" / "liquidity_backtest.json"
    if bt.exists():
        (out / "backtest.json").write_text(json.dumps(json.loads(bt.read_text())["pooled"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    import sys
    from pathlib import Path
    write_site(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
