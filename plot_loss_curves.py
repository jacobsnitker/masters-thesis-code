"""
Training and validation loss curves for ML classifiers that expose per-iteration loss:
  GB      — log loss per boosting round via staged_predict_proba
  LGBM    — log loss per boosting round via eval_set (train + val)
  MLP     — log loss per epoch via warm_start loop
  XGBoost — log loss per boosting round via eval_set (train + val)

LDA, LR, RF, SVM do not expose per-iteration loss in sklearn and are omitted.
80/20 stratified train/val split (val set never seen during training).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    "GB":      "lr = 0.1",
    "LGBM":    "lr = 0.001 (paper)",
    "MLP":     "lr = adaptive",
    "XGBoost": "lr = 0.001 (paper)",
}


def load_cache(path: str):
    cache = np.load(path, allow_pickle=True)
    X = pd.DataFrame(cache["X"], columns=cache["columns"])
    y = cache["y"]
    sc = StandardScaler()
    X_sc = sc.fit_transform(X.values.astype(np.float32))
    return X_sc, y


def get_curves(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    curves = {}  # name -> {"train": [...], "val": [...]}

    # GB — staged_predict_proba gives proba at each round for any set
    print("  Training GB...")
    gb = GradientBoostingClassifier(
        n_estimators=300, loss="log_loss", max_depth=3, random_state=42
    )
    gb.fit(X_tr, y_tr)
    train_ll, val_ll = [], []
    for p_tr, p_val in zip(
        gb.staged_predict_proba(X_tr),
        gb.staged_predict_proba(X_val),
    ):
        train_ll.append(log_loss(y_tr, p_tr))
        val_ll.append(log_loss(y_val, p_val))
    curves["GB"] = {"train": train_ll, "val": val_ll}

    # LGBM — eval_set with both train and val
    print("  Training LGBM...")
    lgbm = LGBMClassifier(
        n_estimators=2000, num_leaves=100, learning_rate=0.001,
        random_state=42, verbose=-1, n_jobs=1,
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_metric="binary_logloss",
        eval_names=["train", "val"],
    )
    res = lgbm.evals_result_
    curves["LGBM"] = {
        "train": res["train"]["binary_logloss"],
        "val":   res["val"]["binary_logloss"],
    }

    # MLP — warm_start loop: one epoch at a time, compute val log loss each step
    print("  Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 10), learning_rate="adaptive",
        max_iter=1, warm_start=True, random_state=42,
    )
    mlp_train, mlp_val = [], []
    best_val, no_improve = float("inf"), 0
    patience = 20
    for _ in range(500):
        mlp.fit(X_tr, y_tr)
        mlp_train.append(mlp.loss_)
        vl = log_loss(y_val, mlp.predict_proba(X_val))
        mlp_val.append(vl)
        if vl < best_val:
            best_val, no_improve = vl, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    curves["MLP"] = {"train": mlp_train, "val": mlp_val}

    # XGBoost — eval_set with both train and val
    print("  Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.001, reg_alpha=0.0001,
        n_jobs=1, random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    res = xgb.evals_result()
    curves["XGBoost"] = {
        "train": res["validation_0"]["logloss"],
        "val":   res["validation_1"]["logloss"],
    }

    return curves


def plot_curves(curves_10s, curves_2s):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (name, color) in enumerate(COLORS.items()):
        for row, (curves, tag) in enumerate([
            (curves_10s, "10s windows"),
            (curves_2s,  "2s windows"),
        ]):
            ax = axes[row][col]
            if name not in curves:
                ax.set_visible(False)
                continue

            tr = curves[name]["train"]
            val = curves[name]["val"]
            x_tr  = np.arange(1, len(tr) + 1)
            x_val = np.arange(1, len(val) + 1)

            ax.plot(x_tr,  tr,  color=color, linewidth=1.2, label="Train")
            ax.plot(x_val, val, color=color, linewidth=1.2,
                    linestyle="--", alpha=0.7, label="Validation")

            ax.set_title(f"{name} — {tag}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Iteration / Epoch", fontsize=8)
            ax.set_ylabel("Log Loss", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, framealpha=0.7)

            gap = val[-1] - tr[-1]
            sign = "+" if gap >= 0 else ""
            ax.annotate(
                f"Train: {tr[-1]:.3f}  Val: {val[-1]:.3f}\nGap: {sign}{gap:.3f}  {LR_INFO[name]}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

    fig.suptitle("Training vs Validation Loss — ML Classifiers", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "loss_curves_ml.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out}")


def main():
    print("Loading 10s dataset...")
    X_10, y_10 = load_cache(os.path.join(RESULTS_DIR, "combined_cache.npz"))
    print("Loading 2s dataset...")
    X_2, y_2   = load_cache(os.path.join(RESULTS_DIR, "combined_cache_2s.npz"))

    print("\nTraining on 10s...")
    curves_10s = get_curves(X_10, y_10)
    print("\nTraining on 2s...")
    curves_2s  = get_curves(X_2, y_2)

    plot_curves(curves_10s, curves_2s)


if __name__ == "__main__":
    main()
