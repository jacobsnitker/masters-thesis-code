"""
Measure inference time (ms per sample) for all models and modality combos.
Uses a single 80/20 stratified split — no CV needed.
Runs for both 10s and 2s window datasets.

Usage:
  python measure_inference_time.py          # ML only (no TF needed)
  python measure_inference_time.py --dl     # include CNN + Transformer (needs TF)
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.models.ml_classifiers import get_classifiers

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COMBO_ORDER = [
    "ECG", "EDA", "EEG", "Gaze",
    "ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze",
    "ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze", "All",
]

MODALITY_PREFIXES = {
    "ECG": "ecg_", "EDA": "eda_", "EEG": "eeg_", "Gaze": "gaze_",
}
MODALITY_COMBOS = {
    "ECG":            ["ECG"],
    "EDA":            ["EDA"],
    "EEG":            ["EEG"],
    "Gaze":           ["Gaze"],
    "ECG+EDA":        ["ECG", "EDA"],
    "ECG+EEG":        ["ECG", "EEG"],
    "ECG+Gaze":       ["ECG", "Gaze"],
    "EDA+EEG":        ["EDA", "EEG"],
    "EDA+Gaze":       ["EDA", "Gaze"],
    "EEG+Gaze":       ["EEG", "Gaze"],
    "ECG+EDA+EEG":    ["ECG", "EDA", "EEG"],
    "ECG+EDA+Gaze":   ["ECG", "EDA", "Gaze"],
    "ECG+EEG+Gaze":   ["ECG", "EEG", "Gaze"],
    "EDA+EEG+Gaze":   ["EDA", "EEG", "Gaze"],
    "All":            ["ECG", "EDA", "EEG", "Gaze"],
}
RAW_KEY = {"ECG": "ecg", "EDA": "eda", "EEG": "eeg", "Gaze": "gaze"}

N_REPEATS = 10   # repeat timing and take median to reduce noise


def select_columns(X: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    prefixes = tuple(MODALITY_PREFIXES[m] for m in modalities)
    cols = [c for c in X.columns if c.startswith(prefixes)]
    return X[cols]


def time_inference(predict_fn, X_test, n_repeats: int = N_REPEATS) -> float:
    """Return median inference time in ms per sample."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X_test)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000 / len(X_test))
    return float(np.median(times))


def measure_dataset(cache_path: str, label: str, run_dl: bool) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Dataset: {label}  ({cache_path})")

    if not os.path.exists(cache_path):
        print(f"  Cache not found: {cache_path}")
        return pd.DataFrame()

    cache    = np.load(cache_path, allow_pickle=True)
    X_all    = pd.DataFrame(cache["X"], columns=cache["columns"])
    y_all    = cache["y"]
    raw_X_all = {k: cache[k] for k in ["ecg", "eda", "eeg", "gaze"]}

    print(f"  {len(y_all)} windows, {X_all.shape[1]} features")

    rows = []

    for combo_name in tqdm(COMBO_ORDER, desc="Combos"):
        modalities = MODALITY_COMBOS[combo_name]
        X_combo    = select_columns(X_all, modalities).values.astype(np.float32)
        raw_combo  = {RAW_KEY[m]: raw_X_all[RAW_KEY[m]] for m in modalities}

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_combo, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        n_test = len(X_te)

        # ── ML classifiers ────────────────────────────────────────────────────
        for name, clf in get_classifiers().items():
            try:
                clf.fit(X_tr_sc, y_tr)
                ms = time_inference(clf.predict, X_te_sc)
                rows.append({"combo": combo_name, "model": name,
                             "ms_per_sample": round(ms, 4), "n_test": n_test})
            except Exception as e:
                tqdm.write(f"  {combo_name} {name} ERROR: {e}")

        # ── DL models ─────────────────────────────────────────────────────────
        if run_dl:
            try:
                import tensorflow as tf
                from src.models.transformer_model import train_transformer, predict_transformer
                from src.models.cnn_model import train_cnn, predict_cnn

                # Transformer
                model = train_transformer(X_tr_sc, y_tr, n_epochs=100)
                ms = time_inference(
                    lambda X: predict_transformer(model, X)[0], X_te_sc
                )
                rows.append({"combo": combo_name, "model": "Transformer",
                             "ms_per_sample": round(ms, 4), "n_test": n_test})
                tf.keras.backend.clear_session()

                # CNN
                raw_tr = {k: v[: len(X_tr)] for k, v in raw_combo.items()}
                raw_te = {k: v[len(X_tr):] for k, v in raw_combo.items()}
                cnn = train_cnn(raw_tr, y_tr, n_epochs=100)
                ms = time_inference(
                    lambda X_raw: predict_cnn(cnn, X_raw)[0],
                    raw_te
                )
                rows.append({"combo": combo_name, "model": "CNN",
                             "ms_per_sample": round(ms, 4), "n_test": n_test})
                tf.keras.backend.clear_session()

            except Exception as e:
                tqdm.write(f"  {combo_name} DL ERROR: {e}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"inference_time_{label.replace(' ', '_')}.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return df


def print_summary(df: pd.DataFrame, label: str):
    if df.empty:
        return
    pivot = df.pivot_table(index="combo", columns="model",
                           values="ms_per_sample", aggfunc="mean")
    pivot = pivot.reindex([c for c in COMBO_ORDER if c in pivot.index])
    print(f"\n=== {label} — Inference time (ms / sample) ===")
    print(pivot.to_string(float_format="{:.4f}".format))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl", action="store_true",
                        help="Include CNN and Transformer (requires TensorFlow)")
    args = parser.parse_args()

    df_10s = measure_dataset(
        os.path.join(RESULTS_DIR, "combined_cache.npz"),
        label="10s", run_dl=args.dl,
    )
    df_2s = measure_dataset(
        os.path.join(RESULTS_DIR, "combined_cache_2s.npz"),
        label="2s", run_dl=args.dl,
    )

    print_summary(df_10s, "10s windows")
    print_summary(df_2s,  "2s windows")

    # Combined comparison
    if not df_10s.empty and not df_2s.empty:
        df_10s["window"] = "10s"
        df_2s["window"]  = "2s"
        combined = pd.concat([df_10s, df_2s])
        out = os.path.join(RESULTS_DIR, "inference_time_combined.csv")
        combined.to_csv(out, index=False)
        print(f"\nCombined saved → {out}")


if __name__ == "__main__":
    main()
