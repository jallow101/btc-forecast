# Liquidity map backtest

Candle-based liquidity map (estimated liquidation levels at 10/25/50/100x + swing stops),
Hyperliquid perp candles for BTC, ETH, SOL, WLD. No settings were tuned on this data.

## 15m  (2026-10-03 to 2026-11-21, 3089 test points)

| Test | Result | 90% range |
|---|---|---|
| A. Heavier side hit first, beyond distance | +1.9 pts | -1.5 pts to +5.2 pts |
| C. Heavy cluster WITH the trend | -0.0 pts | -5.4 pts to +4.5 pts |
| C. Heavy cluster AGAINST the trend | +8.5 pts | -0.5 pts to +18.9 pts |
| B. Sweep & reclaim, 4 bars later (n=119) | +0.00% , win 52% | -0.11% to +0.11% |
| B. Control: plain wick & reclaim (n=847) | +0.03% , win 54% | -0.01% to +0.07% |
| Nearest-side rule accuracy / map rule accuracy | 65% / 64% | |

## 1h  (2026-05-09 to 2026-11-21, 3606 test points)

| Test | Result | 90% range |
|---|---|---|
| A. Heavier side hit first, beyond distance | +3.7 pts | +1.0 pts to +6.2 pts |
| C. Heavy cluster WITH the trend | +8.3 pts | +3.9 pts to +12.4 pts |
| C. Heavy cluster AGAINST the trend | +2.4 pts | -6.4 pts to +10.4 pts |
| B. Sweep & reclaim, 4 bars later (n=138) | +0.26% , win 54% | -0.11% to +0.67% |
| B. Control: plain wick & reclaim (n=654) | -0.08% , win 51% | -0.19% to +0.02% |
| Nearest-side rule accuracy / map rule accuracy | 66% / 64% | |

## 4h  (2024-09-29 to 2026-11-21, 6202 test points)

| Test | Result | 90% range |
|---|---|---|
| A. Heavier side hit first, beyond distance | +1.7 pts | -0.4 pts to +3.8 pts |
| C. Heavy cluster WITH the trend | +6.4 pts | +1.6 pts to +11.0 pts |
| C. Heavy cluster AGAINST the trend | +1.8 pts | -3.3 pts to +6.6 pts |
| B. Sweep & reclaim, 3 bars later (n=186) | -0.52% , win 41% | -1.07% to +0.03% |
| B. Control: plain wick & reclaim (n=656) | -0.20% , win 52% | -0.39% to -0.02% |
| Nearest-side rule accuracy / map rule accuracy | 72% / 71% | |
