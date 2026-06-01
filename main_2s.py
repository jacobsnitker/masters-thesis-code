"""
CLARE Replication Study — 2-second sliding window entry point.

Identical pipeline to main.py, with two differences:
  1. Uses 2s windows with 50% overlap (1s step) instead of 10s windows
  2. Results saved to results_10fold_2s.csv / results_loso_2s.csv

Usage:
  python main_2s.py                        # full pipeline
  python main_2s.py --scheme 10fold
  python main_2s.py --no-dl
  python main_2s.py --combo-start 0 --combo-end 5
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

import sys
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

from src.config import PARTICIPANT_IDS, RESULTS_DIR
from src.feature_extraction_2s import build_combined_dataset_2s
from src.evaluation import run_10fold, run_loso

MODALITY_PREFIXES = {
    "ECG":  "ecg_",
    "EDA":  "eda_",
    "EEG":  "eeg_",
    "Gaze": "gaze_",
}

ALL_MODALITIES = list(MODALITY_PREFIXES.keys())

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


def select_columns(X: pd.DataFrame, modalities: list[str]) -> pd.DataFrame:
    prefixes = tuple(MODALITY_PREFIXES[m] for m in modalities)
    cols = [c for c in X.columns if c.startswith(prefixes)]
    return X[cols]


def main():
    parser = argparse.ArgumentParser(description="CLARE Replication Study — 2s windows")
    parser.add_argument("--modalities", nargs="+", choices=ALL_MODALITIES,
                        default=ALL_MODALITIES)
    parser.add_argument("--scheme", choices=["10fold", "loso", "both"], default="both")
    parser.add_argument("--no-dl", action="store_true")
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--participants", nargs="+", default=None)
    parser.add_argument("--combo-start", type=int, default=0)
    parser.add_argument("--combo-end",   type=int, default=None)
    args = parser.parse_args()

    run_dl = not args.no_dl
    run_ml = not args.no_ml
    gpus   = tf.config.list_physical_devices("GPU")
    device = "gpu" if gpus else "cpu"
    print(f"Device: {'GPU' if gpus else 'CPU'} (TensorFlow — {len(gpus)} GPU(s) visible)")

    participant_ids = args.participants if args.participants else PARTICIPANT_IDS
    print(f"Participants: {participant_ids}")

    # ── Step 1: Build 2s dataset ───────────────────────────────────────────────
    print("\n=== Building 2s sliding window dataset ===")
    cache_path = os.path.join(RESULTS_DIR, "combined_cache_2s.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        cache     = np.load(cache_path, allow_pickle=True)
        X_all     = pd.DataFrame(cache["X"], columns=cache["columns"])
        y_all     = cache["y"]
        subj_all  = cache["subjects"]
        raw_X_all = {k: cache[k] for k in ["ecg", "eda", "eeg", "gaze"]}
    else:
        X_all, raw_X_all, y_all, subj_all = build_combined_dataset_2s(participant_ids)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(cache_path,
                 X=X_all.values, columns=X_all.columns.to_numpy(),
                 y=y_all, subjects=subj_all,
                 **raw_X_all)
        print(f"Dataset saved to {cache_path}")

    print(f"Dataset: {X_all.shape[0]} windows, {X_all.shape[1]} features")
    print(f"Class distribution: Low={int((y_all==0).sum())}, High={int((y_all==1).sum())}")
    print(f"Raw windows: ECG {raw_X_all['ecg'].shape}, EEG {raw_X_all['eeg'].shape}")

    # ── Step 2: Run CV ─────────────────────────────────────────────────────────
    if set(args.modalities) == set(ALL_MODALITIES):
        combo_items = list(MODALITY_COMBOS.items())
        start  = args.combo_start
        end    = args.combo_end if args.combo_end is not None else len(combo_items)
        combos = dict(combo_items[start:end])
        if start > 0 or end < len(combo_items):
            print(f"Running combos [{start}:{end}]: {list(combos.keys())}")
    else:
        key    = "+".join(sorted(args.modalities))
        combos = {key: args.modalities}

    RAW_KEY = {"ECG": "ecg", "EDA": "eda", "EEG": "eeg", "Gaze": "gaze"}

    path_10fold = os.path.join(RESULTS_DIR, "results_10fold_2s.csv")
    path_loso   = os.path.join(RESULTS_DIR, "results_loso_2s.csv")

    def _load_existing(path):
        if not os.path.exists(path):
            return set(), pd.DataFrame()
        try:
            df = pd.read_csv(path, header=0, index_col=[0, 1])
            return set(df.index.get_level_values(0).unique()), df
        except Exception:
            return set(), pd.DataFrame()

    done_10fold, existing_10fold_df = _load_existing(path_10fold)
    done_loso,   existing_loso_df   = _load_existing(path_loso)
    if done_10fold:
        print(f"Resuming 10-fold: skipping {sorted(done_10fold)}")
    if done_loso:
        print(f"Resuming LOSO: skipping {sorted(done_loso)}")

    all_10fold_results = {}
    all_loso_results   = {}

    combo_bar = tqdm(combos.items(), desc="Modality combos", unit="combo",
                     total=len(combos))
    for combo_name, modalities in combo_bar:
        combo_bar.set_postfix_str(combo_name)
        tqdm.write(f"\n{'='*60}\nModality combo: {combo_name}")
        X_combo   = select_columns(X_all, modalities)
        raw_combo = {RAW_KEY[m]: raw_X_all[RAW_KEY[m]] for m in modalities}
        tb_dir    = os.path.join(RESULTS_DIR, "tensorboard_2s") if run_dl else None

        if args.scheme in ("10fold", "both"):
            if combo_name in done_10fold:
                tqdm.write(f"  [SKIP] 10-fold {combo_name} already done")
            else:
                tqdm.write(f"\n--- 10-Fold CV: {combo_name} ---")
                res = run_10fold(X_combo, y_all, raw_X=raw_combo, run_dl=run_dl, run_ml=run_ml,
                                 n_epochs=args.epochs, device=device,
                                 tb_base_dir=tb_dir, combo_name=combo_name)
                all_10fold_results[combo_name] = res
                tqdm.write(res.to_string())
                merged = pd.concat([existing_10fold_df, pd.concat(all_10fold_results, axis=0)])
                merged[~merged.index.duplicated(keep="last")].to_csv(path_10fold)

        if args.scheme in ("loso", "both"):
            if combo_name in done_loso:
                tqdm.write(f"  [SKIP] LOSO {combo_name} already done")
            else:
                tqdm.write(f"\n--- LOSO CV: {combo_name} ---")
                res = run_loso(X_combo, y_all, subj_all, raw_X=raw_combo, run_dl=run_dl, run_ml=run_ml,
                               n_epochs=args.epochs, device=device,
                               tb_base_dir=tb_dir, combo_name=combo_name)
                all_loso_results[combo_name] = res
                tqdm.write(res.to_string())
                merged = pd.concat([existing_loso_df, pd.concat(all_loso_results, axis=0)])
                merged[~merged.index.duplicated(keep="last")].to_csv(path_loso)

        if run_dl:
            tf.keras.backend.clear_session()

    # ── Step 3: Final save ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if all_10fold_results:
        merged = pd.concat([existing_10fold_df, pd.concat(all_10fold_results, axis=0)])
        merged[~merged.index.duplicated(keep="last")].to_csv(path_10fold)
        print(f"\n10-fold results saved to {path_10fold}")

    if all_loso_results:
        merged = pd.concat([existing_loso_df, pd.concat(all_loso_results, axis=0)])
        merged[~merged.index.duplicated(keep="last")].to_csv(path_loso)
        print(f"LOSO results saved to {path_loso}")

    print("\nDone.")


if __name__ == "__main__":
    main()
