"""
Transformer network as described in CLARE paper (Section VI-A).
Implemented in TensorFlow/Keras to match the original paper's framework.

Architecture:
  Early fusion: concatenate all modality feature vectors before input.
  4 Transformer blocks (multi-head attention + FFN + layer norm + residual).
  Prediction head: FC(256) → ReLU → Dropout(0.5) → FC(128) → ReLU → Dropout(0.5) → FC(1) → Sigmoid

Training:
  Optimizer: Adam (lr=0.0001)
  Loss: Binary Cross-Entropy
  Batch: 256, Epochs: 100
"""

import numpy as np
import tensorflow as tf
if not hasattr(tf, 'keras'):
    from tensorflow import keras
    tf.keras = keras
from tqdm import tqdm


# ── Transformer block ─────────────────────────────────────────────────────────
def _transformer_block(
    x, d_model: int, n_heads: int = 4, dim_ff: int = 256, dropout: float = 0.1
):
    """Multi-head self-attention + FFN + residual connections + layer norm."""
    # Self-attention
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
    )(x, x)
    attn_out = tf.keras.layers.Dropout(dropout)(attn_out)
    x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)(x + attn_out)

    # Feed-forward network
    ff = tf.keras.layers.Dense(dim_ff, activation="relu")(x)
    ff = tf.keras.layers.Dense(d_model)(ff)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)(x + ff)

    return x


# ── Full Transformer model ────────────────────────────────────────────────────
def _build_transformer(
    input_dim: int,
    d_model:   int = 128,
    n_blocks:  int = 4,
    n_heads:   int = 4,
    dropout:   float = 0.5,
) -> tf.keras.Model:
    """
    Early-fusion Transformer.
    Input: (B, D_total) feature vector — treated as single patch (seq_len=1).
    """
    inp = tf.keras.Input(shape=(input_dim,), name="features")

    # Project to d_model; treat as single-patch sequence
    x = tf.keras.layers.Dense(d_model)(inp)
    x = tf.keras.layers.Reshape((1, d_model))(x)   # (B, 1, d_model)

    # 4 Transformer blocks
    for _ in range(n_blocks):
        x = _transformer_block(x, d_model=d_model, n_heads=n_heads, dropout=0.1)

    x = tf.keras.layers.Reshape((d_model,))(x)     # (B, d_model)

    # Prediction head
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="transformer")


# ── Training ──────────────────────────────────────────────────────────────────
def train_transformer(
    X_train:    np.ndarray,   # already-concatenated feature matrix (N, D)
    y_train:    np.ndarray,
    d_model:    int = 128,
    n_epochs:   int = 100,
    batch_size: int = 256,
    device:     str = "cpu",  # TF handles device placement automatically
    log_dir:    str | None = None,
) -> tf.keras.Model:
    input_dim = X_train.shape[1]
    model = _build_transformer(input_dim=input_dim, d_model=d_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )

    X = X_train.astype(np.float32)
    y = y_train.astype(np.float32)

    class LogCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                tqdm.write(
                    f"  Transformer epoch {epoch+1}/{n_epochs}"
                    f" — loss: {logs.get('loss', 0):.4f}"
                )

    callbacks = [LogCallback()]
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=False,
        ))

    model.fit(
        X, y,
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        verbose=0,
        validation_split=0.1 if log_dir is not None else 0.0,
        callbacks=callbacks,
    )
    return model


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_transformer(
    model:      tf.keras.Model,
    X:          np.ndarray,
    batch_size: int = 256,
    device:     str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict(X.astype(np.float32), batch_size=batch_size, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)
    probs_2d = np.stack([1 - probs, probs], axis=1)
    return preds, probs_2d
