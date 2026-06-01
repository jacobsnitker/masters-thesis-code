"""
Feature importance for all 8 ML classifiers on 10s and 2s windows.

Importance method per model:
  Tree-based (GB, RF):      feature_importances_  (mean decrease in impurity)
  Boosted trees (LGBM):     feature_importances_  (gain)
  XGBoost:                  booster gain scores
  Linear (LDA, LR):         |coef_|  (absolute coefficient magnitude)
  SVM, MLP:                 permutation importance on a held-out subset
                            (shuffling each feature and measuring F1 drop)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.models.ml_classifiers import get_classifiers

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TOP_N = 20
PERM_SAMPLES = 1000   # subsample size for permutation importance
PERM_REPEATS = 5

MODALITY_COLOR = {
    "ecg":  "#e07b54",
    "eda":  "#5b9bd5",
    "eeg":  "#70ad47",
    "gaze": "#9b59b6",
}


def get_modality(name: str) -> str:
    return name.split("_")[0]


def extract_importance(clf, X_tr, y_tr, X_val, y_val, feature_names, name) -> pd.Series:
    """Return a Series of (feature → importance score), normalised to sum=1."""

    if name in ("GB", "RF"):
        imp = pd.Series(clf.feature_importances_, index=feature_names)

    elif name == "LGBM":
        imp = pd.Series(
            clf.booster_.feature_importance(importance_type="gain"),
            index=feature_names,
        )

    elif name == "XGBoost":
        raw = clf.get_booster().get_score(importance_type="gain")
        idx_map = {f"f{i}": n for i, n in enumerate(feature_names)}
        imp = pd.Series({idx_map[k]: v for k, v in raw.items()})
        imp = imp.reindex(feature_names).fillna(0)

    elif name in ("LDA", "LR"):
        coef = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
        imp = pd.Series(np.abs(coef), index=feature_names)

    else:
        # SVM, MLP — permutation importance on validation subset
        n = min(PERM_SAMPLES, len(X_val))
        idx = np.random.RandomState(42).choice(len(X_val), n, replace=False)
        Xv, yv = X_val[idx], y_val[idx]
        result = permutation_importance(
            clf, Xv, yv,
            n_repeats=PERM_REPEATS,
            scoring=lambda est, X, y: f1_score(y, est.predict(X), average="macro", zero_division=0),
            random_state=42,
            n_jobs=1,
        )
        imp = pd.Series(result.importances_mean, index=feature_names)
        imp = imp.clip(lower=0)   # negative = noise; treat as zero

    total = imp.sum()
    return (imp / total * 100) if total > 0 else imp


def run_dataset(cache_path: str, tag: str):
    print(f"\n{'='*60}")
    print(f"Dataset: {tag}  ({cache_path})")

    if not os.path.exists(cache_path):
        print("  Cache not found.")
        return {}

    cache = np.load(cache_path, allow_pickle=True)
    X_df  = pd.DataFrame(cache["X"], columns=cache["columns"])
    y     = cache["y"]
    feat  = list(X_df.columns)
    print(f"  {len(y)} windows, {len(feat)} features")

    sc = StandardScaler()
    X  = sc.fit_transform(X_df.values.astype(np.float32))

    # 80/20 split: train on 80%, use 20% as val for permutation importance
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classifiers = get_classifiers()
    importances = {}

    for name, clf in tqdm(classifiers.items(), desc=f"  Models ({tag})"):
        try:
            clf.fit(X_tr, y_tr)
            imp = extract_importance(clf, X_tr, y_tr, X_val, y_val, feat, name)
            importances[name] = imp.sort_values(ascending=False)
            tqdm.write(f"    {name}: top feature = {imp.idxmax()}  ({imp.max():.2f}%)")
        except Exception as e:
            tqdm.write(f"    {name} ERROR: {e}")

    return importances


def plot_top_features(importances: dict, tag: str):
    models = list(importances.keys())
    n = len(models)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    axes = axes.flatten()

    for ax, name in zip(axes, models):
        imp = importances[name].head(TOP_N)
        colors = [MODALITY_COLOR.get(get_modality(f), "#aaa") for f in imp.index]
        ax.barh(range(len(imp)), imp.values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance (%)", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linewidth=0.4)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [plt.Rectangle((0,0),1,1,color=c,label=m.upper())
               for m, c in MODALITY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"Top {TOP_N} Features per Model — {tag} windows", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(RESULTS_DIR, f"feature_importance_all_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


def plot_modality_heatmap(importances: dict, tag: str):
    """Heatmap: rows = modalities, cols = models, values = % of gain."""
    modalities = list(MODALITY_COLOR.keys())
    models = list(importances.keys())

    data = pd.DataFrame(index=modalities, columns=models, dtype=float)
    for m, imp in importances.items():
        mod_idx = [get_modality(f) for f in imp.index]
        series = imp.copy()
        series.index = mod_idx
        totals = series.groupby(level=0).sum()
        for mod in modalities:
            data.loc[mod, m] = totals.get(mod, 0.0)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(data.values.astype(float), cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels([m.upper() for m in modalities], fontsize=9)
    plt.colorbar(im, ax=ax, label="Share of total importance (%)")

    for i in range(len(modalities)):
        for j in range(len(models)):
            v = data.iloc[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=7.5, color="black" if v < 50 else "white")

    ax.set_title(f"Modality Contribution per Model — {tag} windows",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"feature_importance_heatmap_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


def save_csv(importances: dict, tag: str):
    rows = []
    for model, imp in importances.items():
        for rank, (feat, val) in enumerate(imp.head(TOP_N).items(), start=1):
            rows.append({"tag": tag, "model": model, "rank": rank,
                         "feature": feat, "modality": get_modality(feat),
                         "importance_pct": round(val, 3)})
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, f"feature_importance_all_{tag}.csv")
    df.to_csv(out, index=False)
    print(f"Saved → {out}")


def main():
    datasets = [
        ("10s", os.path.join(RESULTS_DIR, "combined_cache.npz")),
        ("2s",  os.path.join(RESULTS_DIR, "combined_cache_2s.npz")),
    ]

    for tag, path in datasets:
        importances = run_dataset(path, tag)
        if importances:
            plot_top_features(importances, tag)
            plot_modality_heatmap(importances, tag)
            save_csv(importances, tag)


if __name__ == "__main__":
    main()
