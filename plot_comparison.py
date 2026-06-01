"""
Compare 10s vs 2s window results for overlapping modality combos.

Produces two figures (LOSO and 10-fold), each with one subplot per shared combo.
Each subplot shows grouped bars: one pair (10s / 2s) per model.

Usage:
  python plot_comparison.py                  # F1_macro (default)
  python plot_comparison.py --metric Accuracy
  python plot_comparison.py --metric F1_weighted
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COL_ORDER = ["GB", "LGBM", "LDA", "LR", "MLP", "RF", "SVM", "XGBoost", "CNN", "Transformer"]

COMBO_ORDER = [
    "ECG", "EDA", "EEG", "Gaze",
    "ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze",
    "ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze", "All",
]

COLOR_10S = "#2980b9"   # blue
COLOR_2S  = "#e67e22"   # orange


def load_pivot(path: str, metric: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=0, index_col=[0, 1])
    df.index.names = ["Modality", "Model"]
    pivot = df[metric].unstack("Model")
    combos = [r for r in COMBO_ORDER if r in pivot.index]
    models = [c for c in COL_ORDER if c in pivot.columns]
    return pivot.loc[combos, models]


def plot_comparison(pivot_10s: pd.DataFrame, pivot_2s: pd.DataFrame,
                    metric: str, scheme: str, out_path: str):
    # Only plot combos present in BOTH datasets
    shared_combos = [c for c in COMBO_ORDER
                     if c in pivot_10s.index and c in pivot_2s.index]
    shared_models = [m for m in COL_ORDER
                     if m in pivot_10s.columns and m in pivot_2s.columns]

    n_combos = len(shared_combos)
    if n_combos == 0:
        print(f"No shared combos for {scheme}, skipping.")
        return

    ncols = min(n_combos, 3)
    nrows = (n_combos + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 3.8),
                             sharey=False)
    axes = np.array(axes).flatten() if n_combos > 1 else [axes]

    x = np.arange(len(shared_models))
    bar_w = 0.38

    for ax_idx, combo in enumerate(shared_combos):
        ax = axes[ax_idx]
        vals_10s = pivot_10s.loc[combo, shared_models].values.astype(float)
        vals_2s  = pivot_2s.loc[combo,  shared_models].values.astype(float)

        bars_10s = ax.bar(x - bar_w / 2, vals_10s, bar_w,
                          color=COLOR_10S, label="10 s", alpha=0.88, zorder=3)
        bars_2s  = ax.bar(x + bar_w / 2, vals_2s,  bar_w,
                          color=COLOR_2S,  label="2 s",  alpha=0.88, zorder=3)

        # Value labels on bars
        for bar in bars_10s:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5,
                        color=COLOR_10S, fontweight="bold")
        for bar in bars_2s:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5,
                        color=COLOR_2S, fontweight="bold")

        ax.set_title(combo, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(shared_models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_ylim(0, max(np.nanmax(vals_10s), np.nanmax(vals_2s)) + 8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    # Hide unused subplots
    for ax in axes[n_combos:]:
        ax.set_visible(False)

    scheme_label = "Leave-One-Subject-Out (LOSO)" if scheme == "loso" else "10-Fold"
    fig.suptitle(
        f"CLARE Replication — {scheme_label} CV  |  {metric}  |  10 s vs 2 s windows",
        fontsize=12, fontweight="bold", y=1.01
    )
    note = ("Note: 10-fold CV on 2s windows has data leakage "
            "(overlapping windows span fold boundaries) — LOSO is the trustworthy comparison."
            if scheme == "10fold" else "")
    if note:
        fig.text(0.5, -0.01, note, ha="center", fontsize=7.5, color="#c0392b",
                 style="italic")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="F1_macro",
                        choices=["Accuracy", "F1_weighted", "F1_macro"])
    args = parser.parse_args()
    metric = args.metric

    for scheme, fname_10s, fname_2s in [
        ("loso",   "results_loso.csv",   "results_loso_2s.csv"),
        ("10fold", "results_10fold.csv", "results_10fold_2s.csv"),
    ]:
        p10 = load_pivot(os.path.join(RESULTS_DIR, fname_10s), metric)
        p2s = load_pivot(os.path.join(RESULTS_DIR, fname_2s),  metric)

        if p10 is None:
            print(f"Missing {fname_10s}, skipping {scheme}.")
            continue
        if p2s is None:
            print(f"Missing {fname_2s}, skipping {scheme}.")
            continue

        out = os.path.join(RESULTS_DIR, f"comparison_{scheme}_{metric.lower()}.png")
        plot_comparison(p10, p2s, metric, scheme, out)


if __name__ == "__main__":
    main()
