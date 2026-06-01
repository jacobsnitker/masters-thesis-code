"""
Compare unbalanced vs balanced class-weight LOSO results for ML classifiers.

Plots F1_macro (default), Accuracy, or F1_weighted side by side for each
modality combo, one subplot per combo.

Usage:
  python plot_balanced_comparison.py
  python plot_balanced_comparison.py --metric Accuracy
  python plot_balanced_comparison.py --metric F1_weighted
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COMBO_ORDER = [
    "ECG", "EDA", "EEG", "Gaze",
    "ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze",
    "ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze", "All",
]

ML_ORDER = ["GB", "LGBM", "LDA", "LR", "MLP", "RF", "SVM", "XGBoost"]

COLOR_UNBAL = "#2980b9"   # blue
COLOR_BAL   = "#27ae60"   # green


def load_pivot(path: str, metric: str, models: list[str]) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=0, index_col=[0, 1])
    df.index.names = ["Modality", "Model"]
    pivot = df[metric].unstack("Model")
    combos = [r for r in COMBO_ORDER if r in pivot.index]
    cols   = [c for c in models if c in pivot.columns]
    return pivot.loc[combos, cols]


def plot_balanced_comparison(metric: str, out_path: str):
    unbal = load_pivot(os.path.join(RESULTS_DIR, "results_loso.csv"),          metric, ML_ORDER)
    bal   = load_pivot(os.path.join(RESULTS_DIR, "results_loso_balanced.csv"), metric, ML_ORDER)

    if unbal is None:
        print("Missing results_loso.csv — cannot plot.")
        return
    if bal is None:
        print("Missing results_loso_balanced.csv — run main.py --balanced first.")
        return

    # Only combos present in both
    shared_combos = [c for c in COMBO_ORDER if c in unbal.index and c in bal.index]
    # Only ML models present in both (LDA/MLP may be missing from balanced)
    shared_models = [m for m in ML_ORDER if m in unbal.columns and m in bal.columns]

    if not shared_combos:
        print("No shared combos between the two result files.")
        return

    ncols = min(len(shared_combos), 3)
    nrows = (len(shared_combos) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.5, nrows * 3.8),
                             sharey=False)
    axes = np.array(axes).flatten() if len(shared_combos) > 1 else [axes]

    x     = np.arange(len(shared_models))
    bar_w = 0.38

    for ax_idx, combo in enumerate(shared_combos):
        ax = axes[ax_idx]

        vals_u = unbal.loc[combo, shared_models].values.astype(float)
        vals_b = bal.loc[combo,   shared_models].values.astype(float)

        bars_u = ax.bar(x - bar_w / 2, vals_u, bar_w,
                        color=COLOR_UNBAL, label="Unbalanced", alpha=0.88, zorder=3)
        bars_b = ax.bar(x + bar_w / 2, vals_b, bar_w,
                        color=COLOR_BAL,   label="Balanced",   alpha=0.88, zorder=3)

        for bar in bars_u:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5,
                        color=COLOR_UNBAL, fontweight="bold")
        for bar in bars_b:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6.5,
                        color=COLOR_BAL, fontweight="bold")

        # Delta annotation: mean improvement
        delta = np.nanmean(vals_b) - np.nanmean(vals_u)
        delta_str = f"Δ={delta:+.1f}"
        delta_col = "#27ae60" if delta > 0 else "#c0392b"
        ax.text(0.98, 0.97, delta_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold", color=delta_col)

        ax.set_title(combo, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(shared_models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ymax = max(np.nanmax(vals_u), np.nanmax(vals_b)) + 8
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for ax in axes[len(shared_combos):]:
        ax.set_visible(False)

    fig.suptitle(
        f"CLARE Replication — LOSO CV  |  {metric}  |  Unbalanced vs Balanced class weights",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.text(
        0.5, -0.01,
        "Δ = mean balanced − mean unbalanced per combo  |  "
        "LDA and MLP omitted from balanced (no class-weight support)",
        ha="center", fontsize=7.5, color="#777", style="italic"
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="F1_macro",
                        choices=["Accuracy", "F1_weighted", "F1_macro"])
    args = parser.parse_args()

    out = os.path.join(RESULTS_DIR,
                       f"balanced_comparison_loso_{args.metric.lower()}.png")
    plot_balanced_comparison(args.metric, out)


if __name__ == "__main__":
    main()
