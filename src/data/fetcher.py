"""Download and cache raw price data from Yahoo Finance.

Reads tickers/date range from configs/config.yaml (CLI flags override).
Each symbol is cached to data/raw/<symbol>.csv; cached files are never
re-downloaded. ^VIX is fetched as an extra feature alongside the tickers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

# Project root = two levels up from this file (src/data/fetcher.py).
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load the YAML config into a dict.

    Parameters
    ----------
    path: Path
        Path to config.yaml.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _cache_path(symbol: str, raw_dir: Path) -> Path:
    """Map a symbol to its cache files, stripping the '^' from index tickers."""
    safe = symbol.lstrip("^")
    return raw_dir / f"{safe}.csv"

def fetch_symbol(
    symbol: str,
    start: str,
    end: str,
    raw_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch one symbol's OHLCV history, using the cache when present.
    
    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker (e.g. "AAPL" or "^VIX").
    start, end : str
        Inclusive date range, "YYYY-MM-DD".
    raw_dir : Path
        Directory where CSVs are cached.
    force : bool
        If True, re-download even if a cache file exists.

    Returns
    -------
    pd.DataFrame
        OHLCV indexed by date.
    """
    path = _cache_path(symbol, raw_dir)
    if path.exists() and not force:
        print(f"[cache] {symbol} -> {path.name}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print(f"[fetch] {symbol} {start} -> {end}")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({start} to {end}).")

    # yfinance may return a MultiIndex column header for a single ticker; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df

def fetch_all(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: Path,
    vix_symbol: str | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch every ticker (plus VIX, if given) and return them by symbol."""
    symbols = list(tickers)
    if vix_symbol:
        symbols.append(vix_symbol)
    return {s: fetch_symbol(s, start, end, raw_dir, force=force) for s in symbols}


def main() -> None:
    cfg = load_config()
    d = cfg["data"]

    parser = argparse.ArgumentParser(description="Fetch & cache raw price data")
    parser.add_argument("--tickers", nargs="+", default=d["tickers"])
    parser.add_argument("--start", default=d["start"])
    parser.add_argument("--end", default=d["end"])
    parser.add_argument(
        "--force", action="store_true", help="re-download even if cached"
    )
    args = parser.parse_args()

    raw_dir = ROOT / d["raw_dir"]
    fetch_all(
        args.tickers,
        args.start,
        args.end,
        raw_dir,
        vix_symbol=d.get("vix_symbol"),
        force=args.force,
    )
    print(f"Done. Cached to {raw_dir}")


if __name__ == "__main__":
    main()

