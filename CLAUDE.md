# stock-vol-predictor

Python/TensorFlow project predicting short-horizon realized stock volatility using LSTM and Transformer models on public equity data (Yahoo Finance via `yfinance`).

## Stack

- Python 3.11+
- TensorFlow 2.x / Keras
- yfinance, pandas, numpy, scikit-learn
- PyYAML (config), matplotlib / seaborn (plots)
- pytest (tests)

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Fetch & cache data (runs fast after first call)
python src/data/fetcher.py --tickers SPY AAPL MSFT GOOGL --start 2015-01-01 --end 2024-12-31

# Preprocess (outputs to data/processed/)
python src/data/preprocessor.py

# Train
python src/train.py --model lstm       # or --model transformer
python src/train.py --model both       # trains and compares both

# Evaluate (prints RMSE, QLIKE, MZ-R²)
python src/evaluate.py --model lstm --checkpoint models/lstm_best.keras

# Single-ticker inference
python src/predict.py --ticker AAPL --model lstm

# Tests
pytest tests/ -v

# Notebook (EDA + visualization)
jupyter notebook notebooks/eda.ipynb
```

## Project Structure

```
configs/config.yaml       ← ALL hyperparameters live here; read before touching train.py
data/raw/                 ← cached yfinance CSVs (gitignored)
data/processed/           ← windowed numpy arrays (gitignored)
src/data/fetcher.py       ← download + cache logic
src/data/preprocessor.py  ← feature engineering, normalization, windowing
src/models/lstm.py        ← stacked LSTM architecture
src/models/transformer.py ← encoder-only Transformer architecture
src/train.py              ← training loop, walk-forward CV, early stopping, checkpointing
src/evaluate.py           ← RMSE, QLIKE, Mincer-Zarnowitz evaluation
src/predict.py            ← inference on a single ticker
notebooks/eda.ipynb       ← EDA, correlation heatmaps, vol regime plots
tests/test_pipeline.py    ← shape checks, NaN guards, lookahead-bias assertions
agent_docs/architecture.md ← model specs and walk-forward CV design
agent_docs/features.md    ← feature formulas and lookahead-bias rules
agent_docs/evaluation.md  ← metric definitions and academic context
```

## Key Constraints

- **No lookahead bias.** Features at time `t` use only data through `t-1`. Scaler must be fit on train folds only, never on the full series. Tests in `tests/test_pipeline.py` assert this.
- **Walk-forward validation only.** No random train/test splits on time series. See `agent_docs/architecture.md` for the fold design.
- **All hyperparameters in `configs/config.yaml`.** No magic numbers in model or training files.
- **Cache before fetch.** `fetcher.py` checks `data/raw/<ticker>.csv` before calling yfinance. Never re-download unnecessarily.
- **Gitignore `data/` and `models/`.** Raw data and checkpoints are never committed.

## Code Style

- Type hints on all function signatures.
- NumPy-format docstrings.
- Preprocessing functions must be stateless given a fitted scaler (required for correct inference).

## Do Not Touch

- `data/raw/` and `data/processed/` (runtime artifacts)
- `models/` directory (checkpoints)

## Deeper Specs

Read these files only when working on the relevant subsystem:

- Model architectures → `agent_docs/architecture.md`
- Feature engineering → `agent_docs/features.md`
- Metrics & academic grounding → `agent_docs/evaluation.md`