"""Walk-forward training for the volatility models.

Pools all tickers into one dataset, folds by global calendar date with an
expanding train window, and fits a fresh StandardScaler on each fold's train
rows only (never on test/holdout) to preserve the no-lookahead rule. The last
'final_holdout_months' of data are reserved during CV; the saved checkpoint is
retrained on all non-holdout data. All hyperparameters come from config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# preprocessor.py uses script-relative imports ('from fetcher import ...'),
# so its own directory must be on sys.path when we import it as a module here.
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from data.fetcher import ROOT, load_config
from data.preprocessor import make_windows
from models.lstm import build_lstm
from models.transformer import build_transformer
from metrics import rmse, qlike, mincer_zarnowitz

def set_seeds(seed: int) -> None:
    """Seed numpy + tensorflow for reproducible runs."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_processed(cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """Load every ticker's processed (X, y, dates) from data/processed/.
    
    Returns
    -------
    dict
        ticker -> {"X": (n, f) float32, "y": (n,) float 32,
                    "dates": (n,) datetime64[ns]}.
    """
    proc_dir = ROOT / cfg["data"]["processed_dir"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for ticker in cfg["data"]["tickers"]:
        npz = np.load(proc_dir / f"{ticker}.npz", allow_pickle=True)
        out[ticker] = {
            "X": npz["X"].astype(np.float32),
            "y": npz["y"].astype(np.float32),
            "dates": npz["dates"].astype("datetime64[ns]"),
        }
    return out

def holdout_cutoff(data: dict[str, dict[str, np.ndarray]], cfg: dict) -> pd.Timestamp:
    """First date of the reserved tail = max_date - final_holdout_months."""
    max_date = max(pd.Timestamp(d["dates"].max()) for d in data.values())
    months = cfg["walk_forward"]["final_holdout_months"]
    return max_date - pd.DateOffset(months=months)

def fold_boundaries(
    min_date: pd.Timestamp, holdout_start: pd.Timestamp, cfg: dict
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate (train_end, test_end) date pairs for expanding walk-forward folds.
    
    train_end starts at min_date + min_train_years and steps forward by
    step_months; each test window spands test_months. Folds stop once a test
    window would reach into the held-out tail.
    """
    wf = cfg["walk_forward"]
    train_end = min_date + pd.DateOffset(years=wf["min_train_years"])
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while True:
        test_end = train_end + pd.DateOffset(months=wf["test_months"])
        if test_end > holdout_start:
            break
        folds.append((train_end, test_end))
        train_end = train_end + pd.DateOffset(months=wf["step_months"])
    return folds

def make_fold_data(
    data: dict[str, dict[str, np.ndarray]],
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale (train-fit only) and window one fold, pooling all tickers.

    A window is labeled by the date of its LAST day (the prediction date) and
    assigned to train if that date < train_date, to test if it falls in
    [train_end, test_end]. The scaler is fit ONLY on feature rows dated before
    train_end, then applied to every ticker's full series, so no test_period
    statistics ever leak into training.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
        Windowed arrays of shape (m, seq_len, n_features) and (m,).
    """
    train_end = np.datetime64(train_end)
    test_end = np.datetime64(test_end)

    # 1. Fit one scaler on all train_period rows, pooled across tickers.
    train_rows = [d["X"][d["dates"] < train_end] for d in data.values()]
    scaler = StandardScaler().fit(np.concatenate(train_rows, axis=0))

    # 2. Window each ticker separately (windows must not cross tickers),
    #    then split by each window's label date.
    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
    for d in data.values():
        Xs = scaler.transform(d["X"].astype(np.float32))
        Xw, yw = make_windows(Xs, d["y"], seq_len)
        label_dates = d["dates"][seq_len - 1:] # date of each window's last day

        is_train = label_dates < train_end
        is_test = (label_dates >= train_end) & (label_dates < test_end)

        Xtr_list.append(Xw[is_train])
        ytr_list.append(yw[is_train])
        Xte_list.append(Xw[is_test])
        yte_list.append(yw[is_test])

    X_train = np.concatenate(Xtr_list, axis=0)
    y_train = np.concatenate(ytr_list, axis=0)
    X_test = np.concatenate(Xte_list, axis=0)
    y_test = np.concatenate(yte_list, axis=0)
    return X_train, y_train, X_test, y_test


def test_ticker_index(
    data: dict[str, dict[str, np.ndarray]],
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    seq_len: int,
) -> np.ndarray:
    """Ticker label for each pooled test row of one fold (aligned to make_fold_data)."""
    ts_start, ts_end = np.datetime64(train_end), np.datetime64(test_end)
    labels: list[np.ndarray] = []
    for ticker, d in data.items():
        label_dates = d["dates"][seq_len - 1:]
        is_test = (label_dates >= ts_start) & (label_dates < ts_end)
        labels.append(np.full(int(is_test.sum()), ticker))
    return np.concatenate(labels)

def build_model(name: str, seq_len: int, n_features: int, cfg: dict) -> keras.Model:
    """Dispatch to the requested architecture."""
    if name == "lstm":
        return build_lstm(seq_len, n_features, cfg)
    if name == "transformer":
        return build_transformer(seq_len, n_features, cfg)
    raise ValueError(f"unknown model '{name}' (expected 'lstm' or 'transformer')")

def compile_model(model: keras.Model, cfg: dict) -> None:
    """Attach Adam + Huber loss from config (shared by both architecture)."""
    t = cfg["training"]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=t["learning_rate"]),
        loss=keras.losses.Huber(delta=t["huber_delta"]),
        metrics=["mae"],
    )

def make_callbacks(cfg: dict) -> list[keras.callbacks.Callback]:
    """Early stopping + reduce-on plateau, both watching the val metric."""
    t = cfg["training"]
    return [
        keras.callbacks.EarlyStopping(
            monitor=t["monitor"],
            patience=t["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=t["monitor"],
            factor=t["reduce_lr_factor"],
            patience=t["reduce_lr_patience"],
        ),
    ]

def run_walk_forward(
    data: dict[str, dict[str, np.ndarray]], model_name: str, cfg: dict
) -> list[dict]:
    """Train a fresh model on each expanding fold; report per-fold val metrics."""
    t = cfg["training"]
    seq_len = cfg["windowing"]["seq_len"]
    eps = cfg["evaluation"]["qlikes_eps"]
    tickers = cfg["data"]["tickers"]
    n_features = len(cfg["features"]["columns"])
    min_date = min(pd.Timestamp(d["dates"].min()) for d in data.values())
    folds = fold_boundaries(min_date, holdout_cutoff(data, cfg), cfg)

    results: list[dict] = []
    for i, (train_end, test_end) in enumerate(folds):
        Xtr, ytr, Xte, yte = make_fold_data(data, train_end, test_end, seq_len)
        set_seeds(cfg["seed"])
        model = build_model(model_name, seq_len, n_features, cfg)
        compile_model(model, cfg)
        model.fit(
            Xtr, ytr,
            validation_data=(Xte, yte),
            epochs=t["epochs"],
            batch_size=t["batch_size"],
            shuffle=t["shuffle"],
            callbacks=make_callbacks(cfg),
            verbose=0,
        )
        val = model.evaluate(Xte, yte, verbose=0, return_dict=True)
        pred = model.predict(Xte, verbose=0).ravel()
        tix = test_ticker_index(data, train_end, test_end, seq_len)

        def tavg(fn) -> float:
            return float(np.mean([fn(yte[tix == tk], pred[tix == tk]) for tk in tickers]))

        fold_rmse = tavg(rmse)
        fold_qlike = tavg(lambda a, b: qlike(a, b, eps))
        fold_r2 = tavg(lambda a, b: mincer_zarnowitz(a, b)[0])
        results.append({
            "fold": i, "n_train": len(ytr), "n_test": len(yte),
            "test_end": test_end.date().isoformat(), "y_std": float(yte.std()),
            "rmse": fold_rmse, "qlike": fold_qlike, "mz_r2": fold_r2, **val,
        })
        print(
            f" fold {i:2d} [->{test_end.date()}]: y_std={yte.std():.3f} "
            f"RMSE={fold_rmse:.4f} QLIKE={fold_qlike:.4f} MZ-R2={fold_r2:.3f}"
        )
    return results

def train_final(
    data: dict[str, dict[str, np.ndarray]], model_name: str, cfg: dict
) -> keras.Model:
    """Retrain one model on all non-holdout data (last 6mo kept as val for early stopping)."""
    t = cfg["training"]
    seq_len = cfg["windowing"]["seq_len"]
    n_features = len(cfg["features"]["columns"])
    holdout_start = holdout_cutoff(data, cfg)
    val_start = holdout_start - pd.DateOffset(months=cfg["walk_forward"]["test_months"])
    Xtr, ytr, Xval, yval = make_fold_data(data, val_start, holdout_start, seq_len)

    set_seeds(cfg["seed"])
    model = build_model(model_name, seq_len, n_features, cfg)
    compile_model(model, cfg)
    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        shuffle=t["shuffle"],
        callbacks=make_callbacks(cfg),
        verbose=0,
    )
    return model

def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Walk-forward training")
    parser.add_argument(
        "--model", choices=["lstm", "transformer", "both"], default="lstm"
    )
    args = parser.parse_args()

    set_seeds(cfg["seed"])
    data = load_processed(cfg)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    names = ["lstm", "transformer"] if args.model == "both" else [args.model]
    for name in names:
        print(f"=== {name}: walk-forward CV ===")
        results = run_walk_forward(data, name, cfg)
        mean_loss = float(np.mean([r["loss"] for r in results]))
        mean_mae = float(np.mean([r["mae"] for r in results]))
        print(f"{name}: mean val_loss={mean_loss:.5f} mean val_mae={mean_mae:.5f}")
        mean_rmse = float(np.mean([r["rmse"] for r in results]))
        mean_qlike = float(np.mean([r["qlike"] for r in results]))
        mean_r2 = float(np.mean([r["mz_r2"] for r in results]))
        print(f"{name}: mean over folds RMSE={mean_rmse:.4f} "
              f"QLIKE={mean_qlike:.4f} MZ-R2={mean_r2:.3f}")

        print(f"=== {name}: final retrain on all non-holdout data ===")
        model = train_final(data, name, cfg)
        out = models_dir / f"{name}_best.keras"
        model.save(out)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()