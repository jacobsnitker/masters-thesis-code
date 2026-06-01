"""
Quick train vs test F1 diagnostic using a single 80/20 stratified split.
Shows the train/test gap per model and combo to diagnose overfitting.
No CV needed — runs in minutes locally.

Usage:
  python measure_overfit.py          # 10s windows
  python measure_overfit.py --2s     # 2s windows
  python measure_overfit.py --both   # both
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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


def select_columns(X: pd.DataFrame, modalities: list) -> pd.DataFrame:
    prefixes = tuple(MODALITY_PREFIXES[m] for m in modalities)
    return X[[c for c in X.columns if c.startswith(prefixes)]]


def measure_dataset(cache_path: str, label: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Dataset: {label}  ({cache_path})")

    if not os.path.exists(cache_path):
        print(f"  Cache not found: {cache_path}")
        return pd.DataFrame()

    cache = np.load(cache_path, allow_pickle=True)
    X_all = pd.DataFrame(cache["X"], columns=cache["columns"])
    y_all = cache["y"]
    print(f"  {len(y_all)} windows, {X_all.shape[1]} features")

    from sklearn.metrics import f1_score, accuracy_score

    rows = []
    for combo_name in tqdm(COMBO_ORDER, desc="Combos"):
        modalities = MODALITY_COMBOS[combo_name]
        X_combo = select_columns(X_all, modalities).values.astype(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_combo, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        for name, clf in get_classifiers().items():
            try:
                clf.fit(X_tr_sc, y_tr)

                train_preds = clf.predict(X_tr_sc)
                test_preds  = clf.predict(X_te_sc)

                train_f1 = f1_score(y_tr, train_preds, average="macro", zero_division=0) * 100
                test_f1  = f1_score(y_te, test_preds,  average="macro", zero_division=0) * 100
                train_acc = accuracy_score(y_tr, train_preds) * 100
                test_acc  = accuracy_score(y_te, test_preds)  * 100

                rows.append({
                    "combo":       combo_name,
                    "model":       name,
                    "train_f1":    round(train_f1,  2),
                    "test_f1":     round(test_f1,   2),
                    "gap_f1":      round(train_f1 - test_f1, 2),
                    "train_acc":   round(train_acc, 2),
                    "test_acc":    round(test_acc,  2),
                    "gap_acc":     round(train_acc - test_acc, 2),
                })
            except Exception as e:
                tqdm.write(f"  {combo_name} {name} ERROR: {e}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"overfit_diagnostic_{label}.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return df


def print_summary(df: pd.DataFrame, label: str):
    if df.empty:
        return
    print(f"\n=== {label} — Train/Test F1_macro gap (train - test) ===")
    pivot = df.pivot_table(index="combo", columns="model", values="gap_f1")
    pivot = pivot.reindex([c for c in COMBO_ORDER if c in pivot.index])
    print(pivot.round(1).to_string())

    print(f"\n=== {label} — Mean gap per model ===")
    print(df.groupby("model")["gap_f1"].mean().round(1).sort_values(ascending=False).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--2s",   dest="run_2s",   action="store_true")
    parser.add_argument("--both", dest="run_both",  action="store_true")
    args = parser.parse_args()

    run_10s = not args.run_2s or args.run_both
    run_2s  = args.run_2s or args.run_both

    if run_10s:
        df10 = measure_dataset(
            os.path.join(RESULTS_DIR, "combined_cache.npz"), "10s"
        )
        print_summary(df10, "10s")

    if run_2s:
        df2 = measure_dataset(
            os.path.join(RESULTS_DIR, "combined_cache_2s.npz"), "2s"
        )
        print_summary(df2, "2s")


if __name__ == "__main__":
    main()
