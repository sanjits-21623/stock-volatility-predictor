# Evaluation Reference

All metrics are computed in `src/evaluate.py`. Print all four for every model on every fold.

---

## Metrics

### 1. RMSE
```
RMSE = sqrt( mean( (RV_true - RV_pred)^2 ) )
```
Units: annualized vol (same as the target). Intuitive but penalizes absolute errors; a model predicting in a high-vol regime gets unfairly penalized vs. a model in a calm regime.

### 2. QLIKE (Quasi-Likelihood Loss)
```
QLIKE = mean( RV_true / RV_pred - log(RV_true / RV_pred) - 1 )
```
This is the standard loss function in the volatility forecasting literature (Patton 2011, *Journal of Econometrics*). It is robust to outliers and scale-free — the right loss to report to an academic audience. Lower is better; zero is perfect.

Implement as:
```python
def qlike(rv_true, rv_pred):
    ratio = rv_true / np.clip(rv_pred, 1e-8, None)
    return np.mean(ratio - np.log(ratio) - 1)
```

### 3. Mincer-Zarnowitz R² (MZ-R²)
```
Regress: RV_true = α + β × RV_pred + ε
Report: R², β (should be near 1), α (should be near 0)
```
From Mincer & Zarnowitz (1969). An unbiased, efficient forecast has α≈0, β≈1. On daily equity data, R² > 0.25 is considered meaningful in the literature; R² > 0.40 is strong.

### 4. Direction Accuracy
```
Dir_acc = mean( sign(RV_true_{t} - RV_true_{t-1}) == sign(RV_pred_{t} - RV_pred_{t-1}) )
```
Did the model correctly predict whether vol went up or down? Useful for trading applications. Baseline (random) = 0.50.

---

## Reporting Format

Print a summary table after each evaluation run:

```
Model         | RMSE   | QLIKE  | MZ-R²  | Dir Acc
HAR-RV        | 0.0821 | 0.0312 | 0.38   | 0.553
LSTM          | 0.0764 | 0.0287 | 0.44   | 0.581
Transformer   | 0.0751 | 0.0271 | 0.46   | 0.594
Naïve RW      | 0.0953 | 0.0481 | 0.18   | 0.501
Hist Vol (21d)| 0.0899 | 0.0402 | 0.22   | 0.521
```

---

## Key Academic References

- **Andersen & Bollerslev (1998)**: Definition of realized volatility from high-frequency returns; foundational paper.
- **Corsi (2009)**: HAR-RV model (Heterogeneous Autoregressive model) — the canonical linear baseline for daily vol forecasting. *Journal of Financial Econometrics*.
- **Patton (2011)**: Derivation and justification of QLIKE as the correct loss for evaluating vol forecasts. *Journal of Econometrics*.
- **Gu, Kelly & Xiu (2020)**: ML methods (including neural networks) applied to return prediction; feature set inspiration. *Review of Financial Studies*.
- **Christoffersen & Diebold (2000)**: Volatility and correlation forecasting review.

These references signal to a research audience that the methodology is grounded in the literature, not ad hoc.