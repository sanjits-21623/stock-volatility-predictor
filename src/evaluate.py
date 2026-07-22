from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

# Mirror train.py's path setup so preprocessor's script-relative imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from data.preprocessor import log_returns, build_target
from data.fetcher import ROOT, load_config
from metrics import rmse, qlike, mincer_zarnowitz, direction_accuracy
# Reuse the exact, lookahead-safe fold builder from training. Importing train
# also imports models.transformer, registering its custom serializable layer
# so keras.models.load_model can rebuild the Transformer checkpoint.
from train import load_processed, holdout_cutoff, make_fold_data
import models.transformer  # noqa: F401  (registration side effect; belt-and-suspenders)
# --------------------------------------------------------------------------- #
# Holdout construction
# --------------------------------------------------------------------------- #
def holdout_window(
    data: dict[str, dict[str, np.ndarray]], cfg: dict
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(holdout_start, test_end) covering the entire reserved tail."""
    holdout_start = holdout_cutoff(data, cfg)
    max_date = max(pd.Timestamp(d["dates"].max()) for d in data.values())
    return holdout_start, max_date + pd.Timedelta(days=1) # end-exclusive

def holdout_ticker_index(
    data: dict[str, dict[str, np.ndarray]],
    holdout_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seq_len: int,
) -> np.ndarray:
    """Ticker label for each pooled holdout row, aligned to make_fold_data's y_test.

    Same iteration order + test mask as make_fold_data, so row i of the returned
    array names the ticker that produced y_test[i] / any prediction[i].
    """
    ts_start, ts_end = np.datetime64(holdout_start), np.datetime64(test_end)
    labels: list[np.ndarray] = []
    for ticker, d in data.items():
        label_dates = d["dates"][seq_len - 1:]
        is_test = (label_dates >= ts_start) * (label_dates < ts_end)
        labels.append(np.full(int(is_test.sum()), ticker))
    return np.concatenate(labels)

def holdout_dates(
    data: dict[str, dict[str, np.ndarray]],
    holdout_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seq_len: int,
) -> np.ndarray:
    """Per-row holdout dates, aligned to make_fold_data's y_test / holdout_ticker_index."""
    ts_start, ts_end = np.datetime64(holdout_start), np.datetime64(test_end)
    out: list[np.ndarray] = []
    for d in data.values():
        label_dates = d["dates"][seq_len - 1:]
        is_test = (label_dates >= ts_start) & (label_dates < ts_end)
        out.append(pd.to_datetime(label_dates[is_test]))
    return np.concatenate(out)


def baseline_preds(
    data: dict[str, dict[str, np.ndarray]],
    holdout_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seq_len: int,
    cfg: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aligned holdout targets + naive/hist-vol baseline forecasts.
    
    Mirrors make_fold_data's per-ticker windowing and test mask EXACTLY, so the
    returned y matches make_fold_data's y_test row-for-row (asserted in main).
    Trailing-vol feature columns are un-annualized, so multiply by sqrt(252) to
    match the annualized target.
    """
    cols = cfg["features"]["columns"]
    i_rv5, i_rv21 = cols.index("rv_5d"), cols.index("rv_21d")
    sqrt_ann = np.sqrt(cfg["target"]["annualization"])
    ts_start, ts_end = np.datetime64(holdout_start), np.datetime64(test_end)

    y, naive, histvol = [], [], []
    for d in data.values():
        label_dates = d["dates"][seq_len - 1:] # last day of each window
        is_test = (label_dates >= ts_start) & (label_dates < ts_end)
        y.append(d["y"][seq_len - 1:][is_test])
        naive.append(d["X"][seq_len - 1:, i_rv5][is_test] * sqrt_ann)
        histvol.append(d["X"][seq_len - 1:, i_rv21][is_test] * sqrt_ann)

    y_true = np.concatenate(y)
    return y_true, {
        "Naive RW": np.concatenate(naive),
        "Hist Vol 21d": np.concatenate(histvol),
    }

# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def har_predictions(
    data: dict[str, dict[str, np.ndarray]],
    holdout_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seq_len: int,
    cfg: dict,
) -> np.ndarray:
    """HAR-RV (Corsi 2009) forecasts algined to the NN holdout targets.
    
    Builds daily/weekly/monthly annualized realized-vol regressors from each
    ticker's returns, fits ONE pooled OLS on non-holdout rows, and predicts on
    the exact (ticker, date) points the neural nets are scored on.
    """
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    lag_d, lag_w, lag_m = cfg["evaluation"]["baselines"]["har_rv_lags"] # 1, 5, 22
    sqrt_ann = np.sqrt(cfg["target"]["annualization"])
    ts_start, ts_end = np.datetime64(holdout_start), np.datetime64(test_end)

    frames: dict[str, pd.DataFrame] = {}
    for ticker in data:
        df = pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
        rv2 = log_returns(df["Close"]) ** 2
        comps = pd.DataFrame(index=df.index)
        comps["d"] = np.sqrt(rv2.rolling(lag_d).mean()) * sqrt_ann
        comps["w"] = np.sqrt(rv2.rolling(lag_w).mean()) * sqrt_ann
        comps["m"] = np.sqrt(rv2.rolling(lag_m).mean()) * sqrt_ann
        comps = comps.shift(1) # golden rule: info through t-1
        comps["target"] = build_target(df, cfg) # same annualized fwd-5d std
        frames[ticker] = comps.dropna()

    # Pooled OLS: target = c + b_d*d + b_w*w + b_m*m, fit on non-holdout rows only.
    train = pd.concat([f[f.index < holdout_start] for f in frames.values()])
    A = np.column_stack([np.ones(len(train)), train[["d", "w", "m"]].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, train["target"].to_numpy(), rcond=None)

    # Predict on the SAME holdout points the NN uses (same ticker order + mask).
    preds: list[np.ndarray] = []
    for ticker, d in data.items():
        label_dates = d["dates"][seq_len - 1:]
        hd_dates = label_dates[(label_dates >= ts_start) & (label_dates < ts_end)]
        rows = frames[ticker].reindex(pd.to_datetime(hd_dates))
        assert rows[["d", "w", "m"]].notna().all().all(), f"{ticker}: missing HAR dates"
        preds.append(coef[0] + rows[["d", "w", "m"]].to_numpy() @ coef[1:])
    return np.concatenate(preds)


def report(
    rows: list[tuple[str, np.ndarray]],
    rv_true: np.ndarray,
    tix: np.ndarray,
    cfg: dict,
) -> None:
    """Print per-ticker-averaged metrics plus an MZ-R2 breakdown by ticker."""
    eps = cfg["evaluation"]["qlikes_eps"]
    tickers = cfg["data"]["tickers"]

    def avg(pred: np.ndarray, fn) -> float:
        return float(np.mean([fn(rv_true[tix == t], pred[tix == t]) for t in tickers]))

    header = f"{'Model':<14}| {'RMSE':>7} | {'QLIKE':>7} | {'MZ-R2':>6} | {'Dir Acc':>7}"
    print("Per-ticker average:")
    print(header)
    print("-" * len(header))
    for name, pred in rows:
        print(
            f"{name:<14}| {avg(pred, rmse):>7.4f} | "
            f"{avg(pred, lambda a, b: qlike(a, b, eps)):>7.4f} | "
            f"{avg(pred, lambda a, b: mincer_zarnowitz(a, b)[0]):>6.3f} | "
            f"{avg(pred, direction_accuracy):>7.3f}"
        )

    matrix_header = f"{'Model':<14}" + "".join(f"{t:>8}" for t in tickers)
    print("\nMZ-R2 by ticker:")
    print(matrix_header)
    print("-" * len(matrix_header))
    for name, pred in rows:
        vals = [mincer_zarnowitz(rv_true[tix == t], pred[tix == t])[0] for t in tickers]
        print(f"{name:<14}" + "".join(f"{v:>8.3f}" for v in vals))


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate models on the holdout tail.")
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both")
    args = parser.parse_args()

    data = load_processed(cfg)
    seq_len = cfg["windowing"]["seq_len"]
    holdout_start, test_end = holdout_window(data, cfg)
    tix = holdout_ticker_index(data, holdout_start, test_end, seq_len)

    # Neural-net holdout set: scaler fit on non-holdout only (train_end == holdout_start).
    _, _, X_test, y_test = make_fold_data(data, holdout_start, test_end, seq_len)

    # Baselines, aligned to the same holdout targets.
    y_base, baselines = baseline_preds(data, holdout_start, test_end, seq_len, cfg)
    assert np.allclose(y_test, y_base), "baseline/NN holdout targets misaligned"

    rows: list[tuple[str, np.ndarray]] = []
    names = ["lstm", "transformer"] if args.model == "both" else [args.model]
    for name in names:
        ckpt = ROOT / "models" / f"{name}_best.keras"
        if not ckpt.exists():
            print(f"skip {name}: no checkpoint at {ckpt}")
            continue
        model = keras.models.load_model(ckpt)
        pred = model.predict(X_test, verbose=0).ravel()
        rows.append((name.upper(), pred))

    rows.append(("HAR-RV", har_predictions(data, holdout_start, test_end, seq_len, cfg)))
    rows.extend(baselines.items())
    print(f"\nHoldout: {holdout_start.date()} -> {test_end.date()}  "
          f"(n={len(y_test)}, y_mean={y_test.mean():.3f}, y_std={y_test.std():.3f})")
    report(rows, y_test, tix, cfg)


if __name__ == "__main__":
    main()