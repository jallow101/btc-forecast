"""Do market signals improve the forecast? Walk-forward test, 2021 to now.

For each signal group, the gradient-boosting model is trained with price features only and with
price + that signal, on exactly the same weeks (only weeks where the signal exists), refit yearly
using only data known before each test year. The final ensemble is rebuilt with each variant too.
Writes reports/signals_test.md and reports/signals_test.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from btcf import features, signals  # noqa: E402
from btcf.features import FEATS  # noqa: E402
from btcf.models import QS, baseline_uncond, knn_analog, powerlaw_fit, powerlaw_predict, pl_weight  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

HS = [1, 4, 12, 26]
TEST_YEARS = range(2021, 2027)


def gbm(train, test, cols, h):
    out = []
    for q in QS:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.05,
                                          max_depth=3, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        m.fit(train[cols], train[f"y{h}"])
        out.append(m.predict(test[cols]))
    return np.sort(np.vstack(out).T, axis=1)


def pin(y, Q):
    return float(np.mean([np.mean(np.maximum(q * (y - Q[:, i]), (q - 1) * (y - Q[:, i]))) for i, q in enumerate(QS)]))


def main():
    status = signals.update()
    w = features.build()
    S = signals.weekly_features(w.index)
    w = w.join(S)
    w = w[w["complete"]]
    # control: two columns of pure noise. Any "improvement" they show is luck, and sets the bar a
    # real signal has to clear.
    rng = np.random.default_rng(7)
    w = w.assign(noise_a=rng.normal(size=len(w)), noise_b=rng.normal(size=len(w)).cumsum())
    w.loc[w.index < pd.Timestamp("2018-02-01"), ["noise_a", "noise_b"]] = np.nan   # same start as the real signals
    groups = {f"+ {g}": [c for c in cols if c in w] for g, cols in signals.GROUPS.items()}
    groups["+ all signals"] = [c for cols in signals.GROUPS.values() for c in cols if c in w]
    groups["+ random noise (control)"] = ["noise_a", "noise_b"]
    groups = {k: v for k, v in groups.items() if v}
    res = []
    for name, extra in groups.items():
        for h in HS:
            data = w.dropna(subset=FEATS + extra + [f"y{h}"])          # same weeks for both versions
            acc = {"y": [], "g0": [], "g1": [], "e0": [], "e1": []}
            for Y in TEST_YEARS:
                ys0 = pd.Timestamp(f"{Y}-01-01")
                train = data[data.index + pd.Timedelta(weeks=h) < ys0]
                test = data[(data.index >= ys0) & (data.index < pd.Timestamp(f"{Y+1}-01-01"))]
                if len(train) < 100 or not len(test):
                    continue
                b = np.tile(baseline_uncond(train, h), (len(test), 1))
                k = np.vstack([knn_analog(train, test.loc[t], h)[0] for t in test.index])
                plf = powerlaw_fit(w.loc[w.index < ys0, "lp"])
                pl = np.vstack([powerlaw_predict(plf, t, test.loc[t, "lp"], h) for t in test.index])
                a = pl_weight(h)
                for tag, cols in (("0", FEATS), ("1", FEATS + extra)):
                    g = gbm(train, test, cols, h)
                    acc["g" + tag].append(g)
                    acc["e" + tag].append(np.sort((1 - a) * (b + g + k) / 3 + a * pl, axis=1))
                acc["y"].append(test[f"y{h}"].values)
            if not acc["y"]:
                continue
            y = np.concatenate(acc["y"]); G0, G1, E0, E1 = (np.vstack(acc[k]) for k in ("g0", "g1", "e0", "e1"))
            err = lambda G: float(np.median(np.abs(np.exp(y - G[:, 1]) - 1)) * 100)
            dacc = lambda G: float(np.mean(np.sign(y) == np.sign(G[:, 1])) * 100)
            res.append({"variant": name, "h": h, "n": len(y), "first_test_week": str(data.index[data.index >= pd.Timestamp("2021-01-01")].min().date()),
                        "gbm_pin_price": pin(y, G0), "gbm_pin_signal": pin(y, G1),
                        "ens_pin_price": pin(y, E0), "ens_pin_signal": pin(y, E1),
                        "gbm_err_price": err(G0), "gbm_err_signal": err(G1),
                        "gbm_dir_price": dacc(G0), "gbm_dir_signal": dacc(G1)})
    R = pd.DataFrame(res)
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    json.dump({"status": status, "results": res}, open(out / "signals_test.json", "w"), indent=1)
    lines = ["# Do market signals improve the forecast?", "", "Data fetch status:", ""]
    lines += [f"- **{k}**: {v}" for k, v in status.items()]
    lines += ["", "Walk-forward test 2021–now, refit yearly, price-only vs price + signal on exactly the same weeks.",
              "Pinball scores the whole forecast range: lower is better. Change < 0 means the signal helped.", ""]
    lines += ["| Signal | Horizon (wks) | Weeks | GBM pinball price → +signal | Change | Final-model pinball price → +signal | Change | Typical error % | Direction right % |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in R.sort_values(["h", "variant"]).itertuples():
        c1 = (r.gbm_pin_signal / r.gbm_pin_price - 1) * 100; c2 = (r.ens_pin_signal / r.ens_pin_price - 1) * 100
        lines.append(f"| {r.variant[2:]} | {r.h} | {r.n} | {r.gbm_pin_price:.4f} → {r.gbm_pin_signal:.4f} | {c1:+.1f}% | "
                     f"{r.ens_pin_price:.4f} → {r.ens_pin_signal:.4f} | {c2:+.1f}% | {r.gbm_err_price:.1f} → {r.gbm_err_signal:.1f} | "
                     f"{r.gbm_dir_price:.0f} → {r.gbm_dir_signal:.0f} |")
    (out / "signals_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
