# progress.md — stock-vol-predictor

Coaching build: Sanjit types all code; Claude provides + explains. This file is
the definition of "done" — check items off as they land. Keep it in sync.

## Done = every box below checked

The bounded goal: a complete, tested, honestly-evaluated pipeline where the claim
"the LSTM beats HAR across volatility regimes" is backed by evidence.

- [x] `src/evaluate.py` core — RMSE, QLIKE, Mincer-Zarnowitz (R²/β/α), direction
      accuracy on the 2024 holdout; LSTM/Transformer + Naive-RW + Hist-Vol-21d.
      Verified running.
- [x] `src/evaluate.py` HAR-RV baseline — Corsi (2009), lags 1/5/22, pooled OLS,
      date-aligned to the holdout. Typed in & verified running.
- [x] `src/evaluate.py` per-ticker refactor — `holdout_ticker_index` + rewritten
      `report` (per-ticker-averaged table + MZ-R²-by-ticker matrix) + `y_std` in
      the header. Typed in & verified running.
- [x] `src/train.py` multi-regime metrics — per-fold, per-ticker RMSE/QLIKE/MZ-R²
      via shared `src/metrics.py`. Full run confirmed: MZ-R² spikes to ~0.86 in the
      2020 COVID fold, ~0.25 mean over folds. Also created `src/metrics.py`.
- [x] `tests/test_pipeline.py` — 5 tests (window shapes, NaN guards, shift-once,
      no future leak into features, forward-looking target). All pass.
- [x] `src/predict.py` — single-ticker inference; reconstructs the training
      StandardScaler (train.py doesn't persist it) and forecasts the latest window.
      Both models verified. Future nicety: have train.py joblib.dump the scaler.
- [x] `notebooks/eda.ipynb` — data overview + vol-regime time series, SPY feature
      correlation heatmap, and the signature "forecastability vs vol regime" plot.
      All cells verified; figures render.

**✅ PROJECT COMPLETE — all boxes checked (finished 2026-07-20).**

Estimate at coaching pace: ~4–5 focused sessions. (Actual: done in one long session.)

## Next action

Add the full metric suite (per fold, per ticker) to `train.py`'s walk-forward
loop. The per-fold NN models exist only in memory during training, so this work
belongs in `train.py`, not `evaluate.py`.

## Key finding — do not re-derive

All models (incl. HAR) score low MZ-R² (~0.01–0.08) on the 2024 holdout. This is
**not a bug**: HAR hits R²≈0.32 in 2020 (y_std 0.27) vs ~0.05 in calm 2024
(y_std 0.11). R² = explained ÷ total variance, so a low-vol holdout year suppresses
it for everyone. The LSTM does beat HAR on 2024 (pooled R² 0.082 vs 0.051).
Per-ticker scoring is the honest unit and is *lower* than pooled (pooled is
inflated by cross-ticker spread). → Real skill must be measured across regimes,
which is why the `train.py` multi-regime item exists.

## Out of scope (do not let this block "done")

Raising MZ-R² via new features / tuning / log-vol targets is open-ended research
with a possible real ceiling. Treat as optional and time-boxed, not part of done.
