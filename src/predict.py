from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Mirror train.py's path setup so the preprocesseor's script-relative imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from data.fetcher import ROOT, load_config
from train import load_processed, holdout_cutoff
import models.transformer # noqa: F401 register custom layer for load_model


def training_scaler(
    data: dict[str, dict[str, np.ndarray]], val_start: pd.Timestamp
) -> StandardScaler:
    """Refit the pooled StandardScaler on all rows dated before val_start.
    
    Reproduces exactly the scaler train_final() used for the saved checkpoint,
    so the inference sees input on the same scale the weights were trained on.
    """
    rows = [d["X"][d["dates"] < np.datetime64(val_start)] for d in data.values()]
    return StandardScaler().fit(np.concatenate(rows, axis=0))


def latest_window(
    d: dict[str, np.ndarray], scaler: StandardScaler, seq_len: int
) -> tuple[np.ndarray, pd.Timestamp]:
    """Scale and shape the most recent seq_len days into (1, seq_len, n_features)."""
    Xs = scaler.transform(d["X"].astype(np.float32))
    return Xs[-seq_len:][np.newaxis, :, :], pd.Timestamp(d["dates"][-1])

def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Forecast next-horizon realized vol.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    data = load_processed(cfg)
    if ticker not in data:
        raise SystemExit(f"{ticker} not in processed data {list(data)}")

    seq_len = cfg["windowing"]["seq_len"]
    holdout_start = holdout_cutoff(data, cfg)
    val_start = holdout_start - pd.DateOffset(months=cfg["walk_forward"]["test_months"])
    scaler = training_scaler(data, val_start)

    X, last_date = latest_window(data[ticker], scaler, seq_len)
    model = keras.models.load_model(ROOT / "models" / f"{args.model}_best.keras")
    pred = float(model.predict(X, verbose=0).ravel()[0])

    horizon = cfg["target"]["horizon"]
    print(f"{ticker} [{args.model}] as of {last_date.date()}: "
          f"forecast {horizon}-day-forward annualized vol = {pred:.4f}")


if __name__ == "__main__":
    main()