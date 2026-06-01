"""
XGBoost feature importance (gain) for 10s and 2s windows.
Trains on the full dataset (no CV) — purpose is interpretation, not evaluation.
Uses working hyperparameters rather than the paper's collapsing ones.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODALITY_COLOR = {
    "ecg":  "#e07b54",
    "eda":  "#5b9bd5",
    "eeg":  "#70ad47",
    "gaze": "#9b59b6",
}

TOP_N = 25


def get_modality(feature_name: str) -> str:
    return feature_name.split("_")[0]


def train_and_extract(cache_path: str, label: str):
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}")
        return None, None

    cache = np.load(cache_path, allow_pickle=True)
    X = pd.DataFrame(cache["X"], columns=cache["columns"])
    y = cache["y"]
    print(f"\n{label}: {X.shape[0]} windows, {X.shape[1]} features")
    print(f"  Class distribution: Low={int((y==0).sum())} ({100*(y==0).mean():.1f}%), "
          f"High={int((y==1).sum())} ({100*(y==1).mean():.1f}%)")

    sc = StandardScaler()
    X_sc = sc.fit_transform(X.values.astype(np.float32))

    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        n_jobs=4,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_sc, y)

    importance = pd.Series(
        clf.get_booster().get_score(importance_type="gain"),
        name="gain",
    )
    # get_score uses f0, f1, ... indices — map back to feature names
    idx_to_name = {f"f{i}": name for i, name in enumerate(X.columns)}
    importance.index = [idx_to_name.get(k, k) for k in importance.index]
    importance = importance.sort_values(ascending=False)

    # Modality-level summary
    modality_gain = importance.copy()
    modality_gain.index = [get_modality(f) for f in modality_gain.index]
    modality_total = modality_gain.groupby(level=0).sum()
    modality_pct = (modality_total / modality_total.sum() * 100).round(1)
    print(f"\n  Modality share of total gain ({label}):")
    for mod, pct in modality_pct.sort_values(ascending=False).items():
        print(f"    {mod:6s}: {pct:.1f}%")

    return importance, clf


def plot_importance(imp_10s, imp_2s):
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, (imp, label) in zip(axes, [(imp_10s, "10s windows"), (imp_2s, "2s windows")]):
        top = imp.head(TOP_N)
        colors = [MODALITY_COLOR.get(get_modality(f), "#aaa") for f in top.index]

        bars = ax.barh(range(len(top)), top.values, color=colors,
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Gain)", fontsize=9)
        ax.set_title(f"Top {TOP_N} Features — {label}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    # Shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=m.upper())
        for m, c in MODALITY_COLOR.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(RESULTS_DIR, "feature_importance.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out}")


def plot_modality_share(imp_10s, imp_2s):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, (imp, label) in zip(axes, [(imp_10s, "10s"), (imp_2s, "2s")]):
        mod_idx = [get_modality(f) for f in imp.index]
        mod_gain = imp.copy()
        mod_gain.index = mod_idx
        totals = mod_gain.groupby(level=0).sum().sort_values(ascending=False)
        pcts = totals / totals.sum() * 100
        colors = [MODALITY_COLOR.get(m, "#aaa") for m in pcts.index]
        ax.bar(pcts.index, pcts.values, color=colors, edgecolor="white")
        for i, (m, v) in enumerate(pcts.items()):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Share of Total Gain (%)", fontsize=9)
        ax.set_title(f"Modality Contribution — {label}", fontsize=10, fontweight="bold")
        ax.set_ylim(0, pcts.max() * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "feature_importance_modality.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


def print_comparison_table(imp_10s, imp_2s):
    top10s = imp_10s.head(TOP_N).reset_index()
    top10s.columns = ["feature_10s", "gain_10s"]
    top2s  = imp_2s.head(TOP_N).reset_index()
    top2s.columns  = ["feature_2s",  "gain_2s"]

    print(f"\n{'Rank':<5} {'10s Feature':<35} {'Gain':>10}   {'2s Feature':<35} {'Gain':>10}")
    print("-" * 95)
    for i in range(TOP_N):
        f10 = top10s.iloc[i]["feature_10s"] if i < len(top10s) else ""
        g10 = f"{top10s.iloc[i]['gain_10s']:.1f}" if i < len(top10s) else ""
        f2  = top2s.iloc[i]["feature_2s"]  if i < len(top2s)  else ""
        g2  = f"{top2s.iloc[i]['gain_2s']:.1f}"  if i < len(top2s)  else ""
        print(f"{i+1:<5} {f10:<35} {g10:>10}   {f2:<35} {g2:>10}")


def main():
    imp_10s, _ = train_and_extract(
        os.path.join(RESULTS_DIR, "combined_cache.npz"), "10s"
    )
    imp_2s, _ = train_and_extract(
        os.path.join(RESULTS_DIR, "combined_cache_2s.npz"), "2s"
    )

    if imp_10s is not None and imp_2s is not None:
        print_comparison_table(imp_10s, imp_2s)
        plot_importance(imp_10s, imp_2s)
        plot_modality_share(imp_10s, imp_2s)


if __name__ == "__main__":
    main()
