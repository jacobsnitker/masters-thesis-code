"""
VGG-style 1-D CNN as described in CLARE paper (Section VI-A, Table III).
Implemented in TensorFlow/Keras to match the original paper's framework.

Architecture per modality encoder (format: Conv1D, filter_size, n_filters, stride):
  Block 1: Conv1D(64, 32, stride=1) → Conv1D(64, 32, stride=3) → MaxPool(2)
  Block 2: Conv1D(32, 64, stride=1) → Conv1D(32, 64, stride=3) → MaxPool(2)
  Block 3: Conv1D(17,128, stride=1) → Conv1D(17,128, stride=3) → MaxPool(2)
  Block 4: Conv1D( 7,256, stride=1) → Conv1D( 7,256, stride=3) → MaxPool(2)
  → FC(512) → FC(256)

Multimodal: late fusion — separate encoder per modality, concatenate FC(256) outputs.
Final: FC → Softmax(2)

Input shapes per modality (10s windows) — Keras (B, T, C) channels-last format:
  ECG : (B, 5120, 1)  — 512 Hz × 10 s, 1 channel (LL_RA)
  EDA : (B, 1280, 1)  — 128 Hz × 10 s, 1 channel (EDA_filtered)
  EEG : (B, 2560, 4)  — 256 Hz × 10 s, 4 channels
  Gaze: (B,  500, 1)  —  50 Hz × 10 s, 1 channel (left pupil)

Training:
  Optimizer: AdaDelta (rho=0.95, lr=5e-3)
  Loss: Focal loss (alpha=4.0, gamma=2.0)
  Batch: 256, Epochs: 100
"""

import numpy as np
import tensorflow as tf
if not hasattr(tf, 'keras'):
    from tensorflow import keras
    tf.keras = keras
from tqdm import tqdm


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 4.0, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # y_pred: logits (B, 2), y_true: integer labels (B,)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        pt = tf.exp(-ce)
        return tf.reduce_mean(self.alpha * tf.pow(1.0 - pt, self.gamma) * ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(alpha=self.alpha, gamma=self.gamma)
        return cfg


# ── Conditional MaxPool (safety guard for short sequences) ────────────────────
class ConditionalMaxPool1D(tf.keras.layers.Layer):
    """MaxPool1D that skips pooling if the sequence length is < pool_size."""
    def __init__(self, pool_size: int = 2, strides: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides

    def call(self, x):
        length = tf.shape(x)[1]   # runtime length
        return tf.cond(
            length >= self.pool_size,
            lambda: tf.nn.max_pool1d(
                x, ksize=self.pool_size, strides=self.strides, padding="VALID"
            ),
            lambda: x,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(pool_size=self.pool_size, strides=self.strides)
        return cfg


# ── Modality meta ─────────────────────────────────────────────────────────────
# key → (n_channels, seq_len)
MODALITY_META = {
    "ecg":  (1, 5120),
    "eda":  (1, 1280),
    "eeg":  (4, 2560),
    "gaze": (1,  500),
}


# ── Single-modality encoder ───────────────────────────────────────────────────
def _build_encoder(in_channels: int, seq_len: int, name: str) -> tf.keras.Model:
    """
    Functional Keras model: (B, T, C) → (B, 256).
    Keras Conv1D uses channels-last format: (batch, steps, channels).
    """
    inp = tf.keras.Input(shape=(seq_len, in_channels), name=f"{name}_in")
    x = inp

    # Block 1: kernel=64, filters=32
    x = tf.keras.layers.Conv1D(32, 64, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(32, 64, strides=3, padding="same", activation="relu")(x)
    x = ConditionalMaxPool1D(2, 2)(x)

    # Block 2: kernel=32, filters=64
    x = tf.keras.layers.Conv1D(64, 32, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(64, 32, strides=3, padding="same", activation="relu")(x)
    x = ConditionalMaxPool1D(2, 2)(x)

    # Block 3: kernel=17, filters=128
    x = tf.keras.layers.Conv1D(128, 17, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(128, 17, strides=3, padding="same", activation="relu")(x)
    x = ConditionalMaxPool1D(2, 2)(x)

    # Block 4: kernel=7, filters=256
    x = tf.keras.layers.Conv1D(256, 7, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(256, 7, strides=3, padding="same", activation="relu")(x)
    x = ConditionalMaxPool1D(2, 2)(x)

    # GAP → FC(512) → FC(256)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    return tf.keras.Model(inputs=inp, outputs=x, name=name)


# ── Multimodal CNN (late fusion) ──────────────────────────────────────────────
def _build_multimodal_cnn(present: list[str], shapes: dict[str, tuple],
                          n_classes: int = 2) -> tf.keras.Model:
    """
    One encoder per modality, outputs concatenated then classified.
    present: ordered subset of ['ecg', 'eda', 'eeg', 'gaze'].
    shapes: dict mapping modality key → (n_channels, seq_len) derived from actual data.
    """
    inputs, encoded = [], []
    for key in present:
        n_ch, seq_len = shapes[key]
        enc = _build_encoder(n_ch, seq_len, name=f"enc_{key}")
        inp = tf.keras.Input(shape=(seq_len, n_ch), name=f"input_{key}")
        inputs.append(inp)
        encoded.append(enc(inp))

    x = tf.keras.layers.Concatenate()(encoded) if len(encoded) > 1 else encoded[0]
    # Fig 9: single FC → SoftMax after late fusion
    out = tf.keras.layers.Dense(n_classes, name="logits")(x)

    return tf.keras.Model(inputs=inputs, outputs=out, name="multimodal_cnn")


# ── Array preparation: (N, C, T) channels-first → (N, T, C) channels-last ───
def _prepare_arrays(X: dict[str, np.ndarray]) -> tuple[list[str], list[np.ndarray]]:
    """
    Raw arrays stored as (N, C, T) (PyTorch convention).
    Keras Conv1D needs (N, T, C) — transpose here.
    Returns (present_keys, list_of_float32_arrays).
    """
    present = [k for k in ["ecg", "eda", "eeg", "gaze"] if k in X]
    arrays = []
    for k in present:
        arr = X[k]
        if arr.ndim == 2:            # (N, T) → (N, T, 1)
            arr = arr[:, :, np.newaxis]
        else:                        # (N, C, T) → (N, T, C)
            arr = arr.transpose(0, 2, 1)
        arrays.append(arr.astype(np.float32))
    return present, arrays


# ── Training ──────────────────────────────────────────────────────────────────
def train_cnn(
    X_train:    dict[str, np.ndarray],
    y_train:    np.ndarray,
    n_epochs:   int = 100,
    batch_size: int = 256,
    device:     str = "cpu",   # TF handles device placement automatically
    log_dir:    str | None = None,
) -> tf.keras.Model:
    """
    X_train: dict with keys from {'ecg','eda','eeg','gaze'}.
      - ECG/EDA/Gaze: (N, 1, T) or (N, T)
      - EEG:          (N, 4, T)
    Returns trained Keras model.
    """
    present, arrays = _prepare_arrays(X_train)
    # Derive shapes from actual data so the model works for any window duration
    shapes = {k: (arrays[i].shape[2], arrays[i].shape[1]) for i, k in enumerate(present)}
    model = _build_multimodal_cnn(present, shapes)
    model.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=5e-3, rho=0.95),
        loss=FocalLoss(alpha=4.0, gamma=2.0),
    )

    y = y_train.astype(np.int32)
    X_dict = {f"input_{k}": arrays[j] for j, k in enumerate(present)}

    class TqdmCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.bar = tqdm(total=n_epochs, desc="    CNN", leave=False, unit="ep", ncols=72)
        def on_epoch_end(self, epoch, logs=None):
            self.bar.set_postfix(loss=f"{logs.get('loss', 0):.3f}")
            self.bar.update(1)
        def on_train_end(self, logs=None):
            self.bar.close()

    callbacks = [TqdmCallback()]
    if log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=False,
        ))

    model.fit(
        X_dict, y,
        batch_size=batch_size,
        epochs=n_epochs,
        shuffle=True,
        verbose=0,
        validation_split=0.1 if log_dir is not None else 0.0,
        callbacks=callbacks,
    )
    return model


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_cnn(
    model:      tf.keras.Model,
    X:          dict[str, np.ndarray],
    batch_size: int = 256,
    device:     str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    present, arrays = _prepare_arrays(X)
    X_dict = {f"input_{k}": arrays[j] for j, k in enumerate(present)}
    logits = model.predict(X_dict, batch_size=batch_size, verbose=0)
    probs = tf.nn.softmax(logits).numpy()
    return probs.argmax(axis=1), probs
