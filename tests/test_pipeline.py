from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data"))

from data.fetcher import load_config
from data.preprocessor import log_returns, build_features, build_target, make_windows


@pytest.fixture
def cfg() -> dict:
    return load_config()

def _synthetic(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Deterministic Close/Volume frame for unit tests (no network, no CSVs)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)

def test_make_windows_shapes_and_alignment() -> None:
    n, f, seq_len = 50, 8, 10
    X = np.arange(n * f, dtype=float).reshape(n, f)
    y = np.arange(n, dtype=float)
    Xw, yw = make_windows(X, y, seq_len)
    assert Xw.shape == (n - seq_len + 1, seq_len, f)
    assert yw.shape == (n - seq_len + 1,)
    assert np.array_equal(yw, y[seq_len - 1:]) # label = last day of window
    assert np.array_equal(Xw[0], X[:seq_len])
    assert np.array_equal(Xw[-1], X[n - seq_len:])

def test_processed_arrays_have_no_nan(cfg) -> None:
    proc_dir = ROOT / cfg["data"]["processed_dir"]
    for ticker in cfg["data"]["tickers"]:
        path = proc_dir / f"{ticker}.npz"
        if not path.exists():
            pytest.skip(f"{path} not built")
        npz = np.load(path) # only read numeric X/y; no allow_pickle needed
        assert np.isfinite(npz["X"]).all(), f"{ticker}: non-finite in X"
        assert np.isfinite(npz["y"]).all(), f"{ticker}: non-finite in y"

def test_shift_applied_exactly_once(cfg) -> None:
    df = _synthetic()
    vix = pd.Series(20.0, index=df.index)
    feats = build_features(df, vix, cfg)
    lhs = feats["log_returns"].to_numpy()
    rhs = log_returns(df["Close"]).shift(1).to_numpy() # raw return, shifted ONE day
    mask = ~np.isnan(lhs) & ~np.isnan(rhs) 
    assert np.allclose(lhs[mask], rhs[mask]) # fails if shifted 0 or 2 times

def test_future_prices_do_not_leak_into_past_features(cfg) -> None:
    df = _synthetic()
    vix = pd.Series(20.0, index=df.index)
    base = build_features(df, vix, cfg)

    k = 150
    df2 = df.copy()
    df2.iloc[k, df2.columns.get_loc("Close")] *= 1.5
    pert = build_features(df2, vix, cfg)

    a, b = base.iloc[: k + 1].to_numpy(), pert.iloc[: k + 1].to_numpy()
    both = ~np.isnan(a) * ~np.isnan(b)
    assert np.allclose(a[both], b[both]), "future price leaked into a past feature"

def test_target_is_forward_looking(cfg) -> None:
    df = _synthetic()
    horizon = cfg["target"]["horizon"]
    y = build_target(df, cfg)
    assert y.iloc[-horizon:].isna().all() # no future -> last rows undefined

    k = 100
    df2 = df.copy()
    df2.iloc[k, df2.columns.get_loc("Close")] *= 1.5
    y2 = build_target(df2, cfg)
    assert not np.isclose(y.iloc[k - 3], y2.iloc[k - 3]) # target[t] uses returns t+1..t+h
