"""Stacked LSTM for short-horizon realized-volatility forecasting.

Builds the architecture from agent_docs/architecture.md:

    Input(seq_len, n_features)
      -> LSTM(128, return_sequences=True) -> Dropout
      -> LSTM(64, return_sequences=False) -> Dropout
      -> Dense(32, relu)
      -> Dense(1, linear)  <- predicted RV
      
This model is return UNCOMPILED: train.py attaches the optimizer/loss so the
two model families share one compile path and all training hyperparameters
stay in configs/config.yaml.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm(seq_len: int, n_features: int, cfg: dict) -> keras.Model:
    """Construct the stacked-LSTM volatility model.
    
    Parameters
    ----------
    seq_len : int
        Number of trading days per input window (configs: windowing.seq_len).
    n_features : int
        Feature count per day (len of features.columns; currently 8).
    cfg : dict
        Parsed config; reads the 'lstm' block (units, dropout, dense_units).

    Returns
    -------
    keras.Model
        Uncompiled model mapping (batch, seq_len, n_features) -> (batch, 1).
    """
    lc = cfg["lstm"]
    units_1, units_2 = lc["units"]
    dropout = lc["dropout"]
    dense_units = lc["dense_units"]

    inputs = keras.Input(shape=(seq_len, n_features))

    x = layers.LSTM(units_1, return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(units_2, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="lstm_vol")
