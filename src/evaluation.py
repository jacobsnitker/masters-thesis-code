"""
Evaluation: 10-fold and Leave-One-Subject-Out (LOSO) cross-validation.
Metrics: Accuracy and F1 score (macro).

For ML classifiers: uses the pre-computed handcrafted feature matrix X.
For CNN/Transformer: raw signal windows are needed; the CNN is evaluated
  using the feature matrix as a proxy (Transformer) or raw-signal slices (CNN).
  In this replication, both DL models receive the tabular feature matrix,
  matching the paper's approach of using the same 10-second windows.
"""

import os
import sys
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ModuleNotFoundError:
    tf = None
    _TF_AVAILABLE = False
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from src.models.ml_classifiers import get_classifiers
if _TF_AVAILABLE:
    from src.models.transformer_model import train_transformer, predict_transformer
    from src.models.cnn_model          import train_cnn, predict_cnn

# Classifiers that accept sample_weight in .fit() but have no class_weight
# constructor param (or where we handle it via sample_weight for consistency).
_SW_FIT_CLFS = {"GB", "XGBoost"}


def _scale(X_train: np.ndarray, X_test: np.ndarray):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def _metrics(y_true, y_pred) -> tuple[float, float, float]:
    acc     = accuracy_score(y_true, y_pred) * 100
    f1_w    = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    f1_mac  = f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    return round(acc, 2), round(f1_w, 2), round(f1_mac, 2)


def _slice_raw(raw_X: dict | None, idx: np.ndarray) -> dict | None:
    if raw_X is None:
        return None
    return {k: v[idx] for k, v in raw_X.items()}


def _fit_clf(clf, name: str, X_tr, y_tr, balanced: bool):
    """Fit a classifier, passing sample_weight for those that need it."""
    if balanced and name in _SW_FIT_CLFS:
        sw = compute_sample_weight("balanced", y_tr)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        clf.fit(X_tr, y_tr)


# ── 10-fold CV ────────────────────────────────────────────────────────────────
def run_10fold(
    X: pd.DataFrame,
    y: np.ndarray,
    raw_X: dict | None = None,
    run_dl: bool = True,
    run_ml: bool = True,
    n_epochs: int = 100,
    device: str = "cpu",
    tb_base_dir: str | None = None,
    combo_name: str = "unknown",
    balanced: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with Accuracy and F1 per model.
    raw_X: dict of modality arrays {key: (N,C,T)} for CNN; None → skip CNN.
    balanced: if True, apply class-weight correction to ML classifiers.
    """
    if run_dl and not _TF_AVAILABLE:
        run_dl = False

    X_arr = X.values.astype(np.float32)
    kf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    clfs = get_classifiers(balanced=balanced) if run_ml else {}

    _empty = lambda: {"acc": [], "f1_weighted": [], "f1_macro": [],
                      "train_acc": [], "train_f1_weighted": [], "train_f1_macro": []}
    results = {}
    if run_ml:
        results.update({name: _empty() for name in clfs})
    if run_dl:
        results["Transformer"] = _empty()
        if raw_X is not None:
            results["CNN"] = _empty()

    fold_bar = tqdm(enumerate(kf.split(X_arr, y)), total=10,
                    desc="10-fold CV", unit="fold")
    for fold, (tr_idx, te_idx) in fold_bar:
        X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
        y_tr, y_te = y[tr_idx],     y[te_idx]
        X_tr_sc, X_te_sc = _scale(X_tr, X_te)

        if run_ml:
            clf_bar = tqdm(clfs.items(), desc=f"  Fold {fold+1}",
                           leave=False, unit="clf", ncols=72)
            for name, clf in clf_bar:
                clf_bar.set_postfix_str(name)
                try:
                    _fit_clf(clf, name, X_tr_sc, y_tr, balanced)
                    tr_acc, tr_f1w, tr_f1m = _metrics(y_tr, clf.predict(X_tr_sc))
                    acc, f1w, f1m = _metrics(y_te, clf.predict(X_te_sc))
                    results[name]["acc"].append(acc)
                    results[name]["f1_weighted"].append(f1w)
                    results[name]["f1_macro"].append(f1m)
                    results[name]["train_acc"].append(tr_acc)
                    results[name]["train_f1_weighted"].append(tr_f1w)
                    results[name]["train_f1_macro"].append(tr_f1m)
                    tqdm.write(f"    {name:10s}  train_f1m={tr_f1m:.1f}  test_f1m={f1m:.1f}")
                except Exception as e:
                    tqdm.write(f"    {name} ERROR: {e}")

        means = {n: f"{np.mean(v['acc']):.1f}%" for n, v in results.items() if v["acc"]}
        fold_bar.set_postfix(means, refresh=True)

        if run_dl:
            tqdm.write(f"  [Fold {fold+1}] Training Transformer...")
            tf_log = (os.path.join(tb_base_dir, "10fold", combo_name, "Transformer", f"fold_{fold+1:02d}")
                      if tb_base_dir else None)
            if tf_log:
                os.makedirs(tf_log, exist_ok=True)
            model = train_transformer(X_tr_sc, y_tr, n_epochs=n_epochs, device=device, log_dir=tf_log)
            tr_acc, tr_f1w, tr_f1m = _metrics(y_tr, predict_transformer(model, X_tr_sc, device=device)[0])
            preds, _ = predict_transformer(model, X_te_sc, device=device)
            acc, f1w, f1m = _metrics(y_te, preds)
            results["Transformer"]["acc"].append(acc)
            results["Transformer"]["f1_weighted"].append(f1w)
            results["Transformer"]["f1_macro"].append(f1m)
            results["Transformer"]["train_acc"].append(tr_acc)
            results["Transformer"]["train_f1_weighted"].append(tr_f1w)
            results["Transformer"]["train_f1_macro"].append(tr_f1m)
            tqdm.write(f"    Transformer  train_f1m={tr_f1m:.1f}  test_f1m={f1m:.1f}")

            if raw_X is not None:
                tqdm.write(f"  [Fold {fold+1}] Training CNN...")
                cnn_log = (os.path.join(tb_base_dir, "10fold", combo_name, "CNN", f"fold_{fold+1:02d}")
                           if tb_base_dir else None)
                if cnn_log:
                    os.makedirs(cnn_log, exist_ok=True)
                try:
                    cnn = train_cnn(_slice_raw(raw_X, tr_idx), y_tr,
                                    n_epochs=n_epochs, device=device, log_dir=cnn_log)
                    tr_acc, tr_f1w, tr_f1m = _metrics(y_tr, predict_cnn(cnn, _slice_raw(raw_X, tr_idx), device=device)[0])
                    preds, _ = predict_cnn(cnn, _slice_raw(raw_X, te_idx), device=device)
                    acc, f1w, f1m = _metrics(y_te, preds)
                    results["CNN"]["acc"].append(acc)
                    results["CNN"]["f1_weighted"].append(f1w)
                    results["CNN"]["f1_macro"].append(f1m)
                    results["CNN"]["train_acc"].append(tr_acc)
                    results["CNN"]["train_f1_weighted"].append(tr_f1w)
                    results["CNN"]["train_f1_macro"].append(tr_f1m)
                    tqdm.write(f"    CNN          train_f1m={tr_f1m:.1f}  test_f1m={f1m:.1f}")
                except Exception as e:
                    tqdm.write(f"    CNN ERROR: {e}")

            tf.keras.backend.clear_session()

    summary = {
        name: {
            "Accuracy":          round(float(np.mean(v["acc"])),               2) if v["acc"]               else float("nan"),
            "F1_weighted":       round(float(np.mean(v["f1_weighted"])),       2) if v["f1_weighted"]       else float("nan"),
            "F1_macro":          round(float(np.mean(v["f1_macro"])),          2) if v["f1_macro"]          else float("nan"),
            "Train_Accuracy":    round(float(np.mean(v["train_acc"])),         2) if v["train_acc"]         else float("nan"),
            "Train_F1_weighted": round(float(np.mean(v["train_f1_weighted"])), 2) if v["train_f1_weighted"] else float("nan"),
            "Train_F1_macro":    round(float(np.mean(v["train_f1_macro"])),    2) if v["train_f1_macro"]    else float("nan"),
        }
        for name, v in results.items()
    }
    return pd.DataFrame(summary).T


# ── LOSO CV ───────────────────────────────────────────────────────────────────
def run_loso(
    X: pd.DataFrame,
    y: np.ndarray,
    subject_ids: np.ndarray,
    raw_X: dict | None = None,
    run_dl: bool = True,
    run_ml: bool = True,
    n_epochs: int = 100,
    device: str = "cpu",
    tb_base_dir: str | None = None,
    combo_name: str = "unknown",
    balanced: bool = False,
) -> pd.DataFrame:
    """
    Leave-One-Subject-Out cross-validation.
    raw_X: dict of modality arrays {key: (N,C,T)} for CNN; None → skip CNN.
    balanced: if True, apply class-weight correction to ML classifiers.
    """
    if run_dl and not _TF_AVAILABLE:
        run_dl = False

    X_arr = X.values.astype(np.float32)
    unique_subjects = np.unique(subject_ids)

    clfs = get_classifiers(balanced=balanced) if run_ml else {}

    results = {}
    if run_ml:
        results.update({name: {"acc": [], "f1_weighted": [], "f1_macro": [],
                               "train_acc": [], "train_f1_weighted": [], "train_f1_macro": []}
                        for name in clfs})
    if run_dl:
        results["Transformer"] = {"acc": [], "f1_weighted": [], "f1_macro": [],
                                  "train_acc": [], "train_f1_weighted": [], "train_f1_macro": []}
        if raw_X is not None:
            results["CNN"] = {"acc": [], "f1_weighted": [], "f1_macro": [],
                              "train_acc": [], "train_f1_weighted": [], "train_f1_macro": []}

    subj_bar = tqdm(unique_subjects, desc="LOSO CV", unit="subj")
    for subj in subj_bar:
        te_mask = subject_ids == subj
        tr_mask = ~te_mask
        te_idx  = np.where(te_mask)[0]
        tr_idx  = np.where(tr_mask)[0]

        X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
        y_tr, y_te = y[tr_idx],     y[te_idx]

        if len(np.unique(y_te)) < 2:
            tqdm.write(f"  Subject {subj+1}: skipping (only one class in test set)")
            continue

        X_tr_sc, X_te_sc = _scale(X_tr, X_te)

        if run_ml:
            clf_bar = tqdm(clfs.items(), desc=f"  Subj {subj+1}",
                           leave=False, unit="clf", ncols=72)
            for name, clf in clf_bar:
                clf_bar.set_postfix_str(name)
                try:
                    _fit_clf(clf, name, X_tr_sc, y_tr, balanced)
                    tr_acc, tr_f1w, tr_f1m = _metrics(y_tr, clf.predict(X_tr_sc))
                    acc, f1w, f1m = _metrics(y_te, clf.predict(X_te_sc))
                    results[name]["acc"].append(acc)
                    results[name]["f1_weighted"].append(f1w)
                    results[name]["f1_macro"].append(f1m)
                    results[name]["train_acc"].append(tr_acc)
                    results[name]["train_f1_weighted"].append(tr_f1w)
                    results[name]["train_f1_macro"].append(tr_f1m)
                    tqdm.write(f"    {name:10s}  train_f1m={tr_f1m:.1f}  test_f1m={f1m:.1f}")
                except Exception as e:
                    tqdm.write(f"    {name} ERROR: {e}")

        means = {n: f"{np.mean(v['acc']):.1f}%" for n, v in results.items() if v["acc"]}
        subj_bar.set_postfix(means, refresh=True)

        if run_dl:
            tqdm.write(f"  [Subj {subj+1}] Training Transformer...")
            tf_log = (os.path.join(tb_base_dir, "loso", combo_name, "Transformer", f"subj_{subj+1:02d}")
                      if tb_base_dir else None)
            if tf_log:
                os.makedirs(tf_log, exist_ok=True)
            model = train_transformer(X_tr_sc, y_tr, n_epochs=n_epochs, device=device, log_dir=tf_log)
            tr_acc, tr_f1w, tr_f1m = _metrics(y_tr, predict_transformer(model, X_tr_sc, device=device)[0])
            preds, _ = predict_transformer(model, X_te_sc, device=device)
            acc, f1w, f1m = _metrics(y_te, preds)
            results["Transformer"]["acc"].append(acc)
            results["Transformer"]["f1_weighted"].append(f1w)
            results["Transformer"]["f1_macro"].append(f1m)
            results["Transformer"]["train_acc"].append(tr_acc)
            results["Transformer"]["train_f1_weighted"].append(tr_f1w)
            results["Transformer"]["train_f1_macro"].append(tr_f1m)
            tqdm.write(f"    Transformer  train_f1m={tr_f1m:.1f}  test_f1m={f1m:.1f}")

            if raw_X is not None:
                tqdm.write(f"  [Subj {subj+1}] Training CNN...")
                cnn_log = (os.path.join(tb_base_dir, "loso", combo_name, "CNN", f"subj_{subj+1:02d}")
                           if tb_base_dir else None)
                if cnn_log:
                    os.makedirs(cnn_log, exist_ok=True)
                try:
                    cnn = train_cnn(_slice_raw(raw_X, tr_idx), y_tr,
                                    n_epochs=n_epochs, device=device, log_dir=cnn_log)
                    tr_acc, tr_f1w, tr_f1m = _metrics(y_tr, predict_cnn(cnn, _slice_raw(raw_X, tr_idx), device=device)[0])
                    preds, _ = predict_cnn(cnn, _slice_raw(raw_X, te_idx), device=device)
                    acc, f1w, f1m = _metrics(y_te, preds)
                    results["CNN"]["acc"].append(acc)
                    results["CNN"]["f1_weighted"].append(f1w)
                    results["CNN"]["f1_macro"].append(f1m)
                    results["CNN"]["train_acc"].append(tr_acc)
                    results["CNN"]["train_f1_weighted"].append(tr_f1w)
                    results["CNN"]["train_f1_macro"].append(tr_f1m)
                    tqdm.write(f"    CNN          train_f1m={tr_f1m:.1f}  test_f1m={f1m:.1f}")
                except Exception as e:
                    tqdm.write(f"    CNN ERROR: {e}")

            tf.keras.backend.clear_session()

    summary = {
        name: {
            "Accuracy":          round(float(np.mean(v["acc"])),               2) if v["acc"]               else float("nan"),
            "F1_weighted":       round(float(np.mean(v["f1_weighted"])),       2) if v["f1_weighted"]       else float("nan"),
            "F1_macro":          round(float(np.mean(v["f1_macro"])),          2) if v["f1_macro"]          else float("nan"),
            "Train_Accuracy":    round(float(np.mean(v["train_acc"])),         2) if v["train_acc"]         else float("nan"),
            "Train_F1_weighted": round(float(np.mean(v["train_f1_weighted"])), 2) if v["train_f1_weighted"] else float("nan"),
            "Train_F1_macro":    round(float(np.mean(v["train_f1_macro"])),    2) if v["train_f1_macro"]    else float("nan"),
        }
        for name, v in results.items()
    }
    return pd.DataFrame(summary).T
