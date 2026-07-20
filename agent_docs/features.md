# Feature Engineering Reference

All features below are computed in `src/data/preprocessor.py`.
**The golden rule: a feature at index `t` may only use price/volume data through index `t-1`.**

---

## Feature Set

| Feature         | Formula                                          | Notes                            |
|-----------------|--------------------------------------------------|----------------------------------|
| `log_return`    | `log(Close_t / Close_{t-1})`                    | Core signal                      |
| `rv_5d`         | `std(log_return, window=5)`                      | Short-term vol, shifted by 1     |
| `rv_21d`        | `std(log_return, window=21)`                     | Medium-term vol, shifted by 1    |
| `rv_63d`        | `std(log_return, window=63)`                     | Long-term vol (HAR component)    |
| `volume_zscore` | `(Volume_t - μ_vol) / σ_vol`, rolling 21d       | Liquidity proxy                  |
| `rsi_14`        | Standard RSI, period=14                          | Momentum, bounded [0, 100]       |
| `bb_width`      | `(BB_upper - BB_lower) / BB_mid`, period=20, 2σ | Regime indicator                 |
| `vix`           | CBOE VIX (^VIX via yfinance), if available      | Fear index; use as extra feature |

The `rv_5d`, `rv_21d`, `rv_63d` HAR lags are the most predictive — these mirror the HAR-RV baseline and give the model a strong prior.

---

## Normalization

- `log_return`, `rv_*` columns: StandardScaler (fit on train fold only)
- `rsi_14`: divide by 100 to bound to [0, 1] (no scaler needed)
- `bb_width`, `volume_zscore`: StandardScaler
- Target `RV_t`: **do not normalize** — predict in raw annualized vol units so QLIKE is interpretable

---

## Windowing

After feature construction:

1. Drop any rows with NaN (first 63 rows due to rolling windows)
2. Align target: `y_t = RV_{t+5}` (5-day forward RV)
3. Slice into overlapping windows of length `seq_len`
4. Output shapes: `X.shape = (n_samples, seq_len, n_features)`, `y.shape = (n_samples,)`

---

## Lookahead-Bias Checklist

Before adding any new feature, verify all three:
- [ ] Rolling windows use `min_periods=window` or are padded with NaN at the start (not filled forward from future)
- [ ] The feature is shifted so index `t` reflects information available at market close of day `t-1`
- [ ] Scaler statistics are computed on training data only and applied to val/test

The test in `tests/test_pipeline.py::test_no_lookahead` shuffles the last 10 rows of the raw data to a random future date and asserts that processed features for prior dates are unchanged.