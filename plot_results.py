"""
Reproduce the paper's Table IV / V style:
  - Rows = modality combinations
  - Columns = classifiers
  - Cells = "Accuracy (F1)"
  - Best value per row highlighted in bold/colour
Saves to results/table_10fold.png and results/table_loso.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

ROW_ORDER = [
    "ECG", "EDA", "EEG", "Gaze",
    "ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze",
    "ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze",
    "All",
]
COL_ORDER = ["GB", "LGBM", "LDA", "LR", "MLP", "RF", "SVM", "XGBoost", "CNN", "Transformer"]

# Row group separators (index of first row in next group)
GROUPS = [
    (0,  4,  "Uni-modal"),
    (4,  10, "Bi-modal"),
    (10, 14, "Tri-modal"),
    (14, 15, "All modalities"),
]

# Light grey alternating shades per group
GROUP_BG = ["#f0f4f8", "#e8f0e8", "#f4f0e8", "#ede8f4"]


def load_pivot(path: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, index_col=[0, 1])
    df.index.names = ["Modality", "Model"]
    pivot = df[metric].unstack("Model")
    rows = [r for r in ROW_ORDER if r in pivot.index]
    cols = [c for c in COL_ORDER if c in pivot.columns]
    return pivot.loc[rows, cols]


def plot_paper_table(acc: pd.DataFrame, f1: pd.DataFrame,
                     title: str, out_path: str):
    n_rows, n_cols = acc.shape
    acc_v = acc.values.astype(float)
    f1_v  = f1.values.astype(float)

    # Figure sizing: wide enough for all columns + row labels
    col_w = 1.35   # inches per column
    row_h = 0.42   # inches per row
    lmargin = 1.6  # left margin for row labels
    tmargin = 0.9  # top margin for column headers

    fig_w = lmargin + n_cols * col_w + 0.3
    fig_h = tmargin + n_rows * row_h + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Group background bands ─────────────────────────────────────────────────
    for g_idx, (start, end, _) in enumerate(GROUPS):
        y_top = fig_h - tmargin - start * row_h
        y_bot = fig_h - tmargin - end   * row_h
        rect = mpatches.FancyBboxPatch(
            (0, y_bot), fig_w, y_top - y_bot,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=GROUP_BG[g_idx], zorder=0
        )
        ax.add_patch(rect)

    # ── Column header row background ───────────────────────────────────────────
    hdr_rect = mpatches.FancyBboxPatch(
        (0, fig_h - tmargin), fig_w, tmargin,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#2c3e50", zorder=1
    )
    ax.add_patch(hdr_rect)

    # ── Column headers ─────────────────────────────────────────────────────────
    ax.text(lmargin / 2, fig_h - tmargin / 2, "Modalities",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=2)
    for c_idx, model in enumerate(acc.columns):
        cx = lmargin + (c_idx + 0.5) * col_w
        ax.text(cx, fig_h - tmargin / 2, model,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", zorder=2)

    # ── Grid lines ─────────────────────────────────────────────────────────────
    # Vertical lines
    for c_idx in range(n_cols + 1):
        x = lmargin + c_idx * col_w
        ax.plot([x, x], [0.2, fig_h - tmargin],
                color="#bbbbbb", linewidth=0.5, zorder=1)
    # Left edge
    ax.plot([0, 0], [0.2, fig_h], color="#bbbbbb", linewidth=0.5, zorder=1)
    ax.plot([fig_w - 0.05, fig_w - 0.05], [0.2, fig_h], color="#bbbbbb", linewidth=0.5, zorder=1)

    # Horizontal lines
    for r_idx in range(n_rows + 1):
        y = fig_h - tmargin - r_idx * row_h
        lw = 1.2 if r_idx in [0, 4, 10, 14, 15] else 0.4
        col = "#666666" if r_idx in [0, 4, 10, 14, 15] else "#cccccc"
        ax.plot([0, fig_w - 0.05], [y, y], color=col, linewidth=lw, zorder=2)

    # ── Group labels on far left ───────────────────────────────────────────────
    for g_idx, (start, end, label) in enumerate(GROUPS):
        ymid = fig_h - tmargin - (start + end) / 2 * row_h
        ax.text(0.13, ymid, label,
                ha="center", va="center", fontsize=7.5, color="#555",
                rotation=90, style="italic", zorder=3)
        # thin separator bar
        ax.plot([0.26, 0.26],
                [fig_h - tmargin - start * row_h, fig_h - tmargin - end * row_h],
                color="#aaaaaa", linewidth=1.5, zorder=3)

    # ── Best-per-row mask (highest accuracy) ───────────────────────────────────
    best_mask = acc_v == np.nanmax(acc_v, axis=1, keepdims=True)

    # ── Row labels + cell values ───────────────────────────────────────────────
    for r_idx, modality in enumerate(acc.index):
        cy = fig_h - tmargin - (r_idx + 0.5) * row_h

        # Row label
        ax.text(lmargin - 0.1, cy, modality,
                ha="right", va="center", fontsize=8.2, color="#111", zorder=3)

        for c_idx in enumerate(acc.columns):
            c_idx = c_idx[0]
            a = acc_v[r_idx, c_idx]
            f = f1_v[r_idx, c_idx]
            is_best = best_mask[r_idx, c_idx]
            cx = lmargin + (c_idx + 0.5) * col_w

            # Highlight best cell
            if is_best:
                hi = mpatches.FancyBboxPatch(
                    (lmargin + c_idx * col_w + 0.04,
                     fig_h - tmargin - (r_idx + 1) * row_h + 0.03),
                    col_w - 0.08, row_h - 0.06,
                    boxstyle="round,pad=0.02", linewidth=1.5,
                    edgecolor="#c0392b", facecolor="#fdecea", zorder=2
                )
                ax.add_patch(hi)

            weight = "bold" if is_best else "normal"
            color  = "#c0392b" if is_best else "#111111"

            ax.text(cx, cy + 0.07, f"{a:.2f}",
                    ha="center", va="center", fontsize=8,
                    fontweight=weight, color=color, zorder=4)
            ax.text(cx, cy - 0.1, f"({f:.2f})",
                    ha="center", va="center", fontsize=6.8,
                    color="#555555" if not is_best else "#c0392b",
                    zorder=4)

    # ── Footer note ────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, 0.12,
            "Values: Accuracy (F1)  |  Best per row highlighted in red",
            ha="center", va="center", fontsize=7.5, color="#777", zorder=3)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#111")
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    for fname, cv_label in [("results_10fold.csv", "10-Fold CV"),
                             ("results_loso.csv",   "Leave-One-Subject-Out (LOSO) CV")]:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"Skipping {fname} (not found)")
            continue

        acc   = load_pivot(path, "Accuracy")
        f1w   = load_pivot(path, "F1_weighted")
        f1mac = load_pivot(path, "F1_macro")

        base = fname.replace('.csv', '')

        plot_paper_table(
            acc, f1w,
            title=f"CLARE Replication — {cv_label}  |  Accuracy (Weighted F1)",
            out_path=os.path.join(RESULTS_DIR, f"table_{base}_weighted.png"),
        )
        plot_paper_table(
            acc, f1mac,
            title=f"CLARE Replication — {cv_label}  |  Accuracy (Macro F1)",
            out_path=os.path.join(RESULTS_DIR, f"table_{base}_macro.png"),
        )


if __name__ == "__main__":
    main()
