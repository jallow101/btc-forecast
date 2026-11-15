# Do market signals improve the forecast?

Data fetch status:

- **fng**: ok, 3206 rows to 2026-11-15
- **funding**: failed: HTTP Error 403: Forbidden
- **dxy**: failed: The read operation timed out
- **ry10**: failed: The read operation timed out
- **stable**: ok, 3274 rows to 2026-11-15

Walk-forward test 2021–now, refit yearly, price-only vs price + signal on exactly the same weeks.
Pinball scores the whole forecast range: lower is better. Change < 0 means the signal helped.

| Signal | Horizon (wks) | Weeks | GBM pinball price → +signal | Change | Final-model pinball price → +signal | Change | Typical error % | Direction right % |
|---|---|---|---|---|---|---|---|---|
| all signals | 1 | 306 | 0.0196 → 0.0198 | +0.8% | 0.0188 → 0.0189 | +0.5% | 3.9 → 4.0 | 52 → 50 |
| fng | 1 | 306 | 0.0196 → 0.0194 | -1.4% | 0.0188 → 0.0188 | -0.1% | 3.9 → 3.8 | 52 → 54 |
| random noise (control) | 1 | 306 | 0.0192 → 0.0200 | +3.9% | 0.0187 → 0.0188 | +0.5% | 3.9 → 4.3 | 52 → 51 |
| stable | 1 | 306 | 0.0196 → 0.0198 | +1.1% | 0.0188 → 0.0189 | +0.5% | 3.9 → 4.0 | 52 → 47 |
| all signals | 4 | 303 | 0.0444 → 0.0451 | +1.7% | 0.0393 → 0.0397 | +1.1% | 12.4 → 11.5 | 59 → 55 |
| fng | 4 | 303 | 0.0444 → 0.0429 | -3.3% | 0.0393 → 0.0391 | -0.6% | 12.4 → 11.5 | 59 → 58 |
| random noise (control) | 4 | 303 | 0.0443 → 0.0418 | -5.8% | 0.0392 → 0.0389 | -0.7% | 12.5 → 10.3 | 58 → 57 |
| stable | 4 | 303 | 0.0444 → 0.0459 | +3.4% | 0.0393 → 0.0397 | +0.9% | 12.4 → 12.3 | 59 → 53 |
| all signals | 12 | 295 | 0.0900 → 0.0885 | -1.7% | 0.0730 → 0.0738 | +1.0% | 20.2 → 22.3 | 62 → 55 |
| fng | 12 | 295 | 0.0900 → 0.0877 | -2.6% | 0.0730 → 0.0730 | -0.0% | 20.2 → 21.0 | 62 → 62 |
| random noise (control) | 12 | 295 | 0.0920 → 0.0889 | -3.3% | 0.0732 → 0.0735 | +0.4% | 18.8 → 17.6 | 63 → 63 |
| stable | 12 | 295 | 0.0900 → 0.0932 | +3.5% | 0.0730 → 0.0735 | +0.6% | 20.2 → 21.9 | 62 → 57 |
| all signals | 26 | 281 | 0.1262 → 0.1195 | -5.4% | 0.1163 → 0.1163 | +0.0% | 30.8 → 30.0 | 71 → 63 |
| fng | 26 | 281 | 0.1262 → 0.1236 | -2.1% | 0.1163 → 0.1163 | +0.0% | 30.8 → 31.0 | 71 → 72 |
| random noise (control) | 26 | 281 | 0.1209 → 0.1192 | -1.4% | 0.1163 → 0.1163 | +0.0% | 28.8 → 30.8 | 72 → 72 |
| stable | 26 | 281 | 0.1262 → 0.1215 | -3.7% | 0.1163 → 0.1163 | +0.0% | 30.8 → 29.8 | 71 → 62 |
