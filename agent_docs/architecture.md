# Architecture Reference

## Target Variable

**5-day forward realized volatility**, annualized:

```
RV_t = std(r_{t+1}, r_{t+2}, r_{t+3}, r_{t+4}, r_{t+5}) × √252
```

where `r_t = log(Close_t / Close_{t-1})`.

This is the label Claude should predict. It is always constructed from *future* returns relative to the prediction date, so it must never appear as an input feature. The preprocessor shifts the label by the horizon before windowing.

---

## LSTM Model (`src/models/lstm.py`)

```
Input (seq_len, n_features)
  → LSTM(128, return_sequences=True) → Dropout(0.2)
  → LSTM(64, return_sequences=False) → Dropout(0.2)
  → Dense(32, activation='relu')
  → Dense(1, activation='linear')   ← predicted RV
```

- Loss: Huber (δ=1.0) — more robust than MSE during volatility spikes
- Optimizer: Adam(lr=1e-3), reduce on plateau
- Batch size: 64; shuffle=False (time series)

---

## Transformer Model (`src/models/transformer.py`)

```
Input (seq_len, n_features)
  → Dense(d_model=64)                   ← linear projection
  → SinusoidalPositionalEncoding
  → 4× TransformerEncoderBlock(
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )
  → GlobalAveragePooling1D
  → Dense(32, activation='relu')
  → Dense(1, activation='linear')
```

Each `TransformerEncoderBlock` is: MultiHeadAttention → Add+Norm → FFN → Add+Norm.

---

## Walk-Forward Validation

The dataset is split into expanding windows with no overlap between train and test:

```
Fold 1: train [2015-01 → 2018-12]  |  test [2019-01 → 2019-06]
Fold 2: train [2015-01 → 2019-06]  |  test [2019-07 → 2019-12]
Fold 3: train [2015-01 → 2019-12]  |  test [2020-01 → 2020-06]
...
```

- Step size: 6 months
- Minimum train: 3 years
- Scaler is refit on each fold's training data; never sees test data

Final held-out test set: last 12 months of available data (never touched during model selection).

---

## Sequence Length

Default `seq_len = 60` trading days (~3 months). Configurable in `configs/config.yaml`.

---

## Baselines to Beat

These must be implemented in `src/evaluate.py` for comparison:

1. **HAR-RV** (Corsi 2009): linear regression on daily, weekly, monthly RV lags — the standard academic benchmark for this task
2. **Historical vol**: rolling 21-day std of log returns
3. **Naïve (random walk)**: predict today's RV as tomorrow's RV

If the LSTM/Transformer cannot beat HAR-RV on QLIKE, that should be reported honestly.