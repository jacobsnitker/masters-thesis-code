"""
Loss curves using the same CV splits as the evaluation:
  10-fold : StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  LOSO    : LeaveOneGroupOut on subject IDs

Per-fold val loss curves are averaged → mean ± 1 std band.
Outputs: results/loss_curves_10fold.png  and  results/loss_curves_loso.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COLORS = {
    "GB":      "#e07b54",
    "LGBM":    "#5b9bd5",
    "MLP":     "#70ad47",
    "XGBoost": "#9b59b6",
}

LR_INFO = {
    "GB":      "lr=0.1",
    "LGBM":    "lr=0.001 (paper)",
    "MLP":     "lr=adaptive",
    "XGBoost": "lr=0.001 (paper)",
}


def load_cache(path):
    cache = np.load(path, allow_pickle=True)
    X = pd.DataFrame(cache["X"], columns=cache["columns"])
    sc = StandardScaler()
    X_sc = sc.fit_transform(X.values.astype(np.float32))
    return X_sc, cache["y"], cache["subjects"]


def train_fold(X_tr, y_tr, X_val, y_val, name):
    """Train one model on one fold, return (train_curve, val_curve)."""
    if name == "GB":
        model = GradientBoostingClassifier(
            n_estimators=300, loss="log_loss", max_depth=3, random_state=42
        )
        model.fit(X_tr, y_tr)
        tr, val = [], []
        for p_tr, p_val in zip(
            model.staged_predict_proba(X_tr),
            model.staged_predict_proba(X_val),
        ):
            tr.append(log_loss(y_tr, p_tr, labels=[0, 1]))
            val.append(log_loss(y_val, p_val, labels=[0, 1]))
        return tr, val

    elif name == "LGBM":
        model = LGBMClassifier(
            n_estimators=2000, num_leaves=100, learning_rate=0.001,
            random_state=42, verbose=-1, n_jobs=1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_metric="binary_logloss",
            eval_names=["train", "val"],
        )
        res = model.evals_result_
        return res["train"]["binary_logloss"], res["val"]["binary_logloss"]

    elif name == "MLP":
        model = MLPClassifier(
            hidden_layer_sizes=(100, 10), learning_rate="adaptive",
            max_iter=1, warm_start=True, random_state=42,
        )
        tr, val = [], []
        best_val, no_improve, patience = float("inf"), 0, 20
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(500):
                model.fit(X_tr, y_tr)
                tr.append(model.loss_)
                vl = log_loss(y_val, model.predict_proba(X_val), labels=[0, 1])
                val.append(vl)
                if vl < best_val:
                    best_val, no_improve = vl, 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    break
        return tr, val

    elif name == "XGBoost":
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.001, reg_alpha=0.0001,
            n_jobs=1, random_state=42, eval_metric="logloss", verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        res = model.evals_result()
        return res["validation_0"]["logloss"], res["validation_1"]["logloss"]


def pad_and_average(curves):
    """Pad shorter curves with their last value, return (mean, std)."""
    max_len = max(len(c) for c in curves)
    padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
    return padded.mean(axis=0), padded.std(axis=0)


def get_cv_curves(X, y, subjects, scheme):
    if scheme == "10fold":
        splits = list(
            StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X, y)
        )
    else:
        splits = list(LeaveOneGroupOut().split(X, y, subjects))

    n = len(splits)
    results = {}
    for name in COLORS:
        print(f"    {name} ({n} folds)...", flush=True)
        tr_curves, val_curves = [], []
        for i, (tr_idx, val_idx) in enumerate(splits):
            print(f"      fold {i+1}/{n}", end="\r", flush=True)
            tr, val = train_fold(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], name)
            tr_curves.append(tr)
            val_curves.append(val)
        print(f"      done{' ' * 20}")
        results[name] = (pad_and_average(tr_curves), pad_and_average(val_curves))
    return results


def plot_cv_loss(scheme, curves_10s, curves_2s, outfile):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (name, color) in enumerate(COLORS.items()):
        for row, (curves, tag) in enumerate([
            (curves_10s, "10s windows"),
            (curves_2s,  "2s windows"),
        ]):
            ax = axes[row][col]
            (tr_mean, tr_std), (val_mean, val_std) = curves[name]
            x_tr  = np.arange(1, len(tr_mean) + 1)
            x_val = np.arange(1, len(val_mean) + 1)

            ax.plot(x_tr,  tr_mean,  color=color, lw=1.2, label="Train")
            ax.fill_between(x_tr,  tr_mean - tr_std,  tr_mean + tr_std,
                            alpha=0.15, color=color)
            ax.plot(x_val, val_mean, color=color, lw=1.2, ls="--",
                    alpha=0.85, label="Validation")
            ax.fill_between(x_val, val_mean - val_std, val_mean + val_std,
                            alpha=0.10, color=color)

            ax.set_title(f"{name} — {tag}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Iteration / Epoch", fontsize=8)
            ax.set_ylabel("Log Loss", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, framealpha=0.7)

            gap = val_mean[-1] - tr_mean[-1]
            sign = "+" if gap >= 0 else ""
            ax.annotate(
                f"Train: {tr_mean[-1]:.3f}  Val: {val_mean[-1]:.3f}\n"
                f"Gap: {sign}{gap:.3f}  {LR_INFO[name]}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

    label = "10-Fold CV" if scheme == "10fold" else "Leave-One-Subject-Out CV"
    fig.suptitle(f"Training vs Validation Loss — {label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {outfile}")


def save_checkpoint(curves, path):
    data = {}
    for name, ((tr_mean, tr_std), (val_mean, val_std)) in curves.items():
        data[f"{name}_tr_mean"] = tr_mean
        data[f"{name}_tr_std"]  = tr_std
        data[f"{name}_val_mean"] = val_mean
        data[f"{name}_val_std"]  = val_std
    np.savez(path, **data)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path):
    data = np.load(path)
    curves = {}
    for name in COLORS:
        curves[name] = (
            (data[f"{name}_tr_mean"],  data[f"{name}_tr_std"]),
            (data[f"{name}_val_mean"], data[f"{name}_val_std"]),
        )
    return curves


def main():
    print("Loading caches...")
    X_10, y_10, subj_10 = load_cache(os.path.join(RESULTS_DIR, "combined_cache.npz"))
    X_2,  y_2,  subj_2  = load_cache(os.path.join(RESULTS_DIR, "combined_cache_2s.npz"))

    for scheme in ["loso"]:
        print(f"\n=== {scheme.upper()} ===")

        ckpt_10s = os.path.join(RESULTS_DIR, f"loss_curves_{scheme}_10s.npz")
        ckpt_2s  = os.path.join(RESULTS_DIR, f"loss_curves_{scheme}_2s.npz")

        if os.path.exists(ckpt_10s):
            print("  10s dataset: loading from checkpoint...")
            curves_10s = load_checkpoint(ckpt_10s)
        else:
            print("  10s dataset...")
            curves_10s = get_cv_curves(X_10, y_10, subj_10, scheme)
            save_checkpoint(curves_10s, ckpt_10s)

        if os.path.exists(ckpt_2s):
            print("  2s dataset: loading from checkpoint...")
            curves_2s = load_checkpoint(ckpt_2s)
        else:
            print("  2s dataset...")
            curves_2s  = get_cv_curves(X_2,  y_2,  subj_2,  scheme)
            save_checkpoint(curves_2s, ckpt_2s)

        outfile = os.path.join(RESULTS_DIR, f"loss_curves_{scheme}.png")
        plot_cv_loss(scheme, curves_10s, curves_2s, outfile)


if __name__ == "__main__":
    main()
