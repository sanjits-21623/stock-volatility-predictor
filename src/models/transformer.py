"""Encoder-only Transformer for short-horizon realized-volatility forecasting.

Architecture (agent_docs/architecture.md):

    Input(seq_len, n_features)
      -> Dense(d_model)      <- linear projection 8 -> 64
      -> SinusoidalPositionalEncoding
      -> num_blocks x TransformerEncoderBlock(num_heads, ff_dim, dropout)
      -> GlobalAveragePooling1D
      -> Dense(32, relu)
      -> Dense(1, linear)    <- predicted RV

Returned UNCOMPILED; train.py attaches optimizer/loss (same path as the LSTM).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="svp")
class SinusoidalPositionalEncoding(layers.Layer):
    """Add fixed sinusoidal position signals to a (batch, seq_len, d_model) tensor.
    
    Attention is order-blind, so each timestep's position must be injected
    explicitly. Uses the original Transformer (Vaswani 2017) sinusoids; they are
    fixed (not learned) and registered as serializable so predict.py can reload
    a saved model.

    Parameters
    ----------
    seq_len : int
        Number of timesteps (configs: windowing.seq_len).
    d_model : int
        Embedding width the encoding is added to.
    """

    def __init__(self, seq_len: int, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self._build_encoding(seq_len, d_model)

    @staticmethod
    def _build_encoding(seq_len: int, d_model: int) -> tf.Tensor:
        pos = np.arange(seq_len)[:, None] # (seq_len, 1)
        i = np.arange(d_model)[None, :] # (1, d_model)
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates # (seq_len, d_model)
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2]) # even dims -> sin
        pe[:, 1::2] = np.cos(angles[:, 1::2]) # odd dims -> cos
        return tf.constant(pe[None, ...]) # (1, seq_len, d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.pos_encoding

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config

def transformer_encoder_block(
    x: tf.Tensor,
    num_heads: int,
    d_model: int, 
    ff_dim: int, 
    dropout: float,
    ) -> tf.Tensor:
    """One encoder block: self-attention -> Add+Norm -> FFN -> Add+Norm.
        
    Parameters
    ----------
    x : tf.Tensor
        Input of shape (batch, seq_len, d_model).
    num_heads : int
        Number of attention heads; key_dim per head is d_model // num_heads.
    d_model : int
        Model width (also the residual-stream width).
    ff_dim : int
        Hidden width of the position-wise feed-forward network.
    dropout : float 
        Dropout rate inside attention and the FFN.

    Returns
    -------
    tf.Tensor 
        Output of shape (batch, seq_len, d_model).
    """
    # 1. Multi self-attention, then residual add + layer norm.
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # 2. Position-wise feed-forward, then residual add + layer norm.
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    return x 

def build_transformer(seq_len: int, n_features: int, cfg: dict) -> keras.Model:
    """Construct the encoder-only Transformer volatility model.
    
    Parameters
    ----------
    seq_len : int
        Trading days per input window (configs: windowing.seq_len).
    n_features : int
        Feature count per day (len of features.columns; currently 8).
    cfg : dict
        Parsed config; reads the 'transformer' block.

    Returns
    -------
    keras.Model
        Uncompiled model mapping (batch, seq_len, n_features) -> (batch, 1).
    """
    tc = cfg["transformer"]
    d_model = tc["d_model"]
    num_blocks = tc["num_blocks"]
    num_heads = tc["num_heads"]
    ff_dim = tc["ff_dim"]
    dropout = tc["dropout"]
    dense_units = tc["dense_units"]

    inputs = keras.Input(shape=(seq_len, n_features))
    x = layers.Dense(d_model)(inputs) # 8 -> 64 projection
    x = SinusoidalPositionalEncoding(seq_len, d_model)(x) # inject position

    for _ in range(num_blocks):
        x = transformer_encoder_block(x, num_heads, d_model, ff_dim, dropout)

    
    x = layers.GlobalAveragePooling1D()(x) # (batch, d_model())
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="transformer_vol")


