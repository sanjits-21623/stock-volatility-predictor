"""Feature engineering, target construction, and windowing

Reads cached CSVs from data/raw/, builds the feature set from
agent_docs/features.md, constructs the 5-day-forward realized-vol target,
and saves one unscaled per-day feature table per ticker to data/processed/.

LOOKAHEAD SAFETY
----------------
Every feature is computed from data through day t, then the ENTIRE feature
frame is shifted by one row (shift(1)) exactly once, so row t holds only
information available at the close of day t-1. Scaling is deliberately NOT
done here: the StandardScaler must be fit on each walk-forward train fold
only, so train.py scales and windows per fold using make_windows() below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fetcher import ROOT, load_config # same dir on sys.path when run as a script


def log_returns(close: pd.Series) -> pd.Series:
    """log(Close_t / Close_{t-1})."""
    return np.log(close / close.shift(1))

def realized_vol(rets: pd.Series, window: int) -> pd.Series:
    """Rolling std of log returns over 'window' days (not yet annualized)."""
    return rets.rolling(window).std()

def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI, bounded [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of volume: (V_t - μ) / σ over `window` days."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std

def bollinger_width(close: pd.Series, period: int, n_std: float) -> pd.Series:
    """(upper - lower) / mid = 2·n_std·σ / mid for a `period`-day band."""
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return (2 * n_std * sd) / mid

def build_features(df: pd.DataFrame, vix: pd.Series, cfg: dict ) -> pd.DataFrame:
    """Assemble the model feature frame, then apply the one-shot shift(1)."""
    f = cfg["features"]
    rets = log_returns(df["Close"])
    short, med, long = f["rv_window"]

    feats = pd.DataFrame(index=df.index)
    feats["log_returns"] = rets
    feats["rv_5d"] = realized_vol(rets, short)
    feats["rv_21d"] = realized_vol(rets, med)
    feats["rv_63d"] = realized_vol(rets, long)
    feats["volume_zscore"] = volume_zscore(df["Volume"], f["volume_zscore_window"])
    feats["rsi_14"] = rsi(df["Close"], f["rsi_period"]) / 100.0 # bound to [0, 1]
    feats["bb_width"] = bollinger_width(df["Close"], f["bb_period"], f["bb_std"])
    feats["vix"] = vix.reindex(df.index)

    # GOLDEN RULE: shift everything by 1 so row t uses only data through t-1
    # Applied exactly once, here. Never shift again downstream. 
    feats = feats.shift(1)

    return feats[f["columns"]] # canonical column order the models expect


def build_target(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """y_t = std(r_{t+1..t+horizon}) · sqrt(annualization)  — forward-looking."""
    t = cfg["target"]
    horizon = t["horizon"]
    rets = log_returns(df["Close"])
    fwd = rets.rolling(horizon).std().shift(-horizon)
    return fwd * np.sqrt(t["annualization"])


def make_windows(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice aligned (X, y) into overlapping sequences.
    
    Returns Xw of shape (n - seq_len + 1, seq_len, n_features) and yw of
    shape (n - seq_len + 1), where yw[i] is the target on the LAST day of
    window i (the window spans days i .. i+seq_len-1).
    """
    n = X.shape[0]
    Xw = np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])
    yw = y[seq_len - 1 :]
    return Xw, yw


def process_ticker(
    ticker: str, vix: pd.Series, cfg: dict, raw_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build features + target for one ticker and drop warmup/tail NaNs."""
    df = pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
    feats = build_features(df, vix, cfg)

    data = feats.copy()
    data["target"] = build_target(df, cfg)
    data = data.dropna() # drops ~63 warmup rows AND the last 'horizon' rows

    X = data[cfg["features"]["columns"]].to_numpy()
    y = data["target"].to_numpy(dtype=np.float32)
    dates = data.index.strftime("%Y-%m-%d").to_numpy()
    return X, y, dates


def main() -> None:
    cfg = load_config()
    d = cfg["data"]

    parser = argparse.ArgumentParser(description="Build processed feature tables.")
    parser.add_argument("--tickers", nargs="+", default=d["tickers"])
    args = parser.parse_args()

    raw_dir = ROOT / d["raw_dir"]
    proc_dir = ROOT / d["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    vix_file = raw_dir / f"{d['vix_symbol'].lstrip('^')}.csv"
    vix = pd.read_csv(vix_file, index_col=0, parse_dates=True)["Close"]

    for ticker in args.tickers:
        X, y, dates = process_ticker(ticker, vix, cfg, raw_dir)
        out = proc_dir / f"{ticker}.npz"
        np.savez(
            out,
            X=X,
            y=y,
            dates=dates,
            feature_names=np.array(cfg["features"]["columns"]),
        )
        print(f"{ticker}: X={X.shape} y={y.shape} -> {out.name}")


if __name__ == "__main__":
    main()