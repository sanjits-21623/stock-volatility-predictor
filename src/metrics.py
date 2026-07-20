from __future__ import annotations

import numpy as np


def rmse(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    """Root mean squared error, in annualized-vol units."""
    return float(np.sqrt(np.mean((rv_true - rv_pred) ** 2)))

def qlike(rv_true: np.ndarray, rv_pred: np.ndarray, eps: float) -> float:
    """Patton (2011) QLIKE loss: mean(r - log r - 1), r = RV_true / RV_pred."""
    ratio = rv_true / np.clip(rv_pred, eps, None)
    return float(np.mean(ratio - np.log(ratio) - 1.0))

def mincer_zarnowitz(rv_true: np.ndarray, rv_pred: np.ndarray) -> tuple[float, float, float]:
    """Fit RV_true = alpha + beta * RV_pred; return (R2, beta, alpha)."""
    beta, alpha = np.polyfit(rv_pred, rv_true, deg=1)
    fitted = alpha + beta * rv_pred
    ss_res = np.sum((rv_true - fitted) ** 2)
    ss_tot = np.sum((rv_true - np.mean(rv_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return float(r2), float(beta), float(alpha)

def direction_accuracy(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    """Fraction of steps where predicted vol moved the same direction as truth."""
    return float(np.mean(np.sign(np.diff(rv_true)) == np.sign(np.diff(rv_pred))))
