"""
Render inference time as a publication-style table (like CLARE Table VI).
Shows 10s and 2s window results side by side.
DL rows shown as '—' if not yet available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODEL_ORDER = [
    "GB", "LGBM", "LDA", "LR", "MLP", "RF", "SVM", "XGBoost",
    "CNN", "Transformer",
]
MODEL_LABELS = {
    "GB":          "Gradient Boosting",
    "LGBM":        "Light Gradient Boosting Machine",
    "LDA":         "Linear Discriminant Analysis",
    "LR":          "Logistic Regression",
    "MLP":         "Multi-Layer Perceptron",
    "RF":          "Random Forest",
    "SVM":         "Support Vector Machine",
    "XGBoost":     "Extreme Gradient Boosting",
    "CNN":         "CNN",
    "Transformer": "Transformer",
}

PAPER_VALUES = {
    "GB":          20.99,
    "LGBM":        64.17,
    "LDA":         15.97,
    "LR":          16.47,
    "MLP":         20.52,
    "RF":          605.88,
    "SVM":         446.64,
    "XGBoost":     1482.30,
    "CNN":         4519.99,
    "Transformer": 4579.37,
}


def load_means(path: str) -> dict:
    """Return mean ms/sample per model averaged across all combos, in μs."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    means = df.groupby("model")["ms_per_sample"].mean() * 1000  # ms → μs
    return means.to_dict()


def plot_table(out_path: str):
    means_10s = load_means(os.path.join(RESULTS_DIR, "inference_time_10s.csv"))
    means_2s  = load_means(os.path.join(RESULTS_DIR, "inference_time_2s.csv"))

    n_rows  = len(MODEL_ORDER)
    col_w   = [3.2, 1.5, 1.5, 1.5]   # label, 10s, 2s, paper
    lpad    = 0.15
    row_h   = 0.42
    hdr_h   = 0.70
    fig_w   = sum(col_w) + lpad * 2
    fig_h   = hdr_h + n_rows * row_h + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Header ────────────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, fig_h - hdr_h), fig_w, hdr_h,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#1a252f", zorder=1))

    col_centers = [
        lpad + col_w[0] / 2,
        lpad + col_w[0] + col_w[1] / 2,
        lpad + col_w[0] + col_w[1] + col_w[2] / 2,
        lpad + col_w[0] + col_w[1] + col_w[2] + col_w[3] / 2,
    ]

    ax.text(col_centers[0], fig_h - hdr_h * 0.38, "Classifiers",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=2)
    ax.text(col_centers[1], fig_h - hdr_h * 0.30, "10s windows",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#7fb3d3", zorder=2)
    ax.text(col_centers[2], fig_h - hdr_h * 0.30, "2s windows",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#82e0aa", zorder=2)
    ax.text(col_centers[3], fig_h - hdr_h * 0.30, "CLARE paper",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#f0b27a", zorder=2)
    for cx in col_centers[1:]:
        ax.text(cx, fig_h - hdr_h * 0.68, "Inference Time (μs)",
                ha="center", va="center", fontsize=7, color="#aaaaaa", zorder=2)

    # ── Column dividers ────────────────────────────────────────────────────────
    x_divs = [
        lpad + col_w[0],
        lpad + col_w[0] + col_w[1],
        lpad + col_w[0] + col_w[1] + col_w[2],
    ]
    for x in x_divs:
        ax.plot([x, x], [0.15, fig_h - hdr_h], color="#cccccc", lw=0.6, zorder=2)

    # ── Thick divider between ML and DL ───────────────────────────────────────
    dl_sep_y = fig_h - hdr_h - 8 * row_h
    ax.plot([0, fig_w], [dl_sep_y, dl_sep_y], color="#555555", lw=1.2, zorder=3)

    # ── Rows ──────────────────────────────────────────────────────────────────
    for r_idx, model in enumerate(MODEL_ORDER):
        cy   = fig_h - hdr_h - (r_idx + 0.5) * row_h
        bg   = "#f7f9fc" if r_idx % 2 == 0 else "white"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, fig_h - hdr_h - (r_idx + 1) * row_h), fig_w, row_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bg, zorder=0))

        # Label
        ax.text(lpad + 0.1, cy, MODEL_LABELS[model],
                ha="left", va="center", fontsize=8.5, color="#111111", zorder=3)

        # 10s value
        val_10s = means_10s.get(model)
        txt_10s = f"{val_10s:,.2f}" if val_10s is not None else "—"
        ax.text(col_centers[1], cy, txt_10s,
                ha="center", va="center", fontsize=8.5, color="#1a5276", zorder=3)

        # 2s value
        val_2s = means_2s.get(model)
        txt_2s = f"{val_2s:,.2f}" if val_2s is not None else "—"
        ax.text(col_centers[2], cy, txt_2s,
                ha="center", va="center", fontsize=8.5, color="#1e8449", zorder=3)

        # Paper value
        paper_val = PAPER_VALUES.get(model)
        txt_paper = f"{paper_val:,.2f}" if paper_val is not None else "—"
        ax.text(col_centers[3], cy, txt_paper,
                ha="center", va="center", fontsize=8.5, color="#784212", zorder=3)

    # ── Border lines ──────────────────────────────────────────────────────────
    ax.plot([0, fig_w], [fig_h - hdr_h, fig_h - hdr_h], color="#333", lw=1.5, zorder=4)
    ax.plot([0, fig_w], [0.15, 0.15], color="#333", lw=1.5, zorder=4)

    # ── Row separators ────────────────────────────────────────────────────────
    for r_idx in range(1, n_rows):
        y = fig_h - hdr_h - r_idx * row_h
        ax.plot([0, fig_w], [y, y], color="#e0e0e0", lw=0.4, zorder=2)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title("Average Inference Time per Window",
                 fontsize=11, fontweight="bold", pad=6, color="#111")

    # ── Footer ────────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, 0.06,
            "CNN and Transformer inference times pending HPC run  |  "
            "averaged across all 15 modality combinations",
            ha="center", va="center", fontsize=6.5, color="#888", zorder=3)

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    out = os.path.join(RESULTS_DIR, "table_inference_time.png")
    plot_table(out)
