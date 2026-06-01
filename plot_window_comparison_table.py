"""
Side-by-side table: 10s vs 2s LOSO results.
One row per modality combo, one column pair per model (10s | 2s).
Metric: F1_macro. Best per row highlighted in red.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COMBO_ORDER = [
    "ECG", "EDA", "EEG", "Gaze",
    "ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze",
    "ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze", "All",
]
ALL_ORDER = ["GB", "LGBM", "LDA", "LR", "MLP", "RF", "SVM", "XGBoost", "CNN", "Transformer"]

GROUPS = [
    (0,  4,  "Uni-modal"),
    (4,  10, "Bi-modal"),
    (10, 14, "Tri-modal"),
    (14, 15, "All"),
]
GROUP_BG = ["#f0f4f8", "#e8f0e8", "#f4f0e8", "#ede8f4"]


def load_pivot(path, metric, model_order):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=0, index_col=[0, 1])
    df.index.names = ["Modality", "Model"]
    pivot = df[metric].unstack("Model")
    combos = [r for r in COMBO_ORDER if r in pivot.index]
    cols   = [c for c in model_order if c in pivot.columns]
    return pivot.loc[combos, cols]


def plot_table(s10: pd.DataFrame, s2: pd.DataFrame, metric: str, title: str, out_path: str):
    models  = s10.columns.tolist()
    s2      = s2.reindex(columns=models)
    combos  = COMBO_ORDER
    n_rows  = len(combos)
    n_models = len(models)

    sub_w  = 0.58
    col_w  = sub_w * 2 + 0.10
    lmargin = 1.7
    tmargin = 1.05
    row_h   = 0.40

    fig_w = lmargin + n_models * col_w + 0.3
    fig_h = tmargin + n_rows * row_h + 0.7

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Group backgrounds ─────────────────────────────────────────────────────
    for g_idx, (start, end, _) in enumerate(GROUPS):
        y_top = fig_h - tmargin - start * row_h
        y_bot = fig_h - tmargin - end   * row_h
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y_bot), fig_w, y_top - y_bot,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=GROUP_BG[g_idx], zorder=0))

    # ── Header ────────────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, fig_h - tmargin), fig_w, tmargin,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#2c3e50", zorder=1))

    ax.text(lmargin / 2, fig_h - tmargin * 0.35, "Modalities",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=2)
    ax.text(lmargin / 2, fig_h - tmargin * 0.72,
            metric.replace("_", " "), ha="center", va="center",
            fontsize=7, color="#aaaaaa", zorder=2)

    for c_idx, model in enumerate(models):
        cx = lmargin + (c_idx + 0.5) * col_w
        ax.text(cx, fig_h - tmargin * 0.32, model,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=2)
        ax.text(cx - sub_w / 2, fig_h - tmargin * 0.72, "10s",
                ha="center", va="center", fontsize=6.5, color="#7fb3d3", zorder=2)
        ax.text(cx + sub_w / 2, fig_h - tmargin * 0.72, "2s",
                ha="center", va="center", fontsize=6.5, color="#82e0aa", zorder=2)

    # ── Grid ──────────────────────────────────────────────────────────────────
    for c_idx in range(n_models + 1):
        x = lmargin + c_idx * col_w
        ax.plot([x, x], [0.2, fig_h - tmargin], color="#bbbbbb", lw=0.5, zorder=1)
    ax.plot([0, 0], [0.2, fig_h], color="#bbbbbb", lw=0.5, zorder=1)
    ax.plot([fig_w - 0.05]*2, [0.2, fig_h], color="#bbbbbb", lw=0.5, zorder=1)
    for r_idx in range(n_rows + 1):
        y  = fig_h - tmargin - r_idx * row_h
        lw = 1.2 if r_idx in [0, 4, 10, 14, 15] else 0.4
        c  = "#666" if r_idx in [0, 4, 10, 14, 15] else "#ccc"
        ax.plot([0, fig_w - 0.05], [y, y], color=c, lw=lw, zorder=2)
    for c_idx in range(n_models):
        x = lmargin + c_idx * col_w + sub_w + 0.05
        ax.plot([x, x], [0.2, fig_h - tmargin],
                color="#dddddd", lw=0.4, linestyle="--", zorder=1)

    # ── Group labels ──────────────────────────────────────────────────────────
    for g_idx, (start, end, label) in enumerate(GROUPS):
        ymid = fig_h - tmargin - (start + end) / 2 * row_h
        ax.text(0.13, ymid, label, ha="center", va="center",
                fontsize=7, color="#555", rotation=90, style="italic", zorder=3)
        ax.plot([0.26]*2,
                [fig_h - tmargin - start * row_h, fig_h - tmargin - end * row_h],
                color="#aaa", lw=1.5, zorder=3)

    # ── Best-per-row mask ─────────────────────────────────────────────────────
    # Build full value arrays including NaN for missing combos
    v10 = np.full((n_rows, n_models), np.nan)
    v2  = np.full((n_rows, n_models), np.nan)
    for r_idx, combo in enumerate(combos):
        if combo in s10.index:
            v10[r_idx] = s10.loc[combo].values.astype(float)
        if combo in s2.index:
            v2[r_idx]  = s2.loc[combo].values.astype(float)

    all_vals = np.concatenate([v10, v2], axis=1)
    row_max  = np.nanmax(all_vals, axis=1, keepdims=True)

    # ── Cells ─────────────────────────────────────────────────────────────────
    for r_idx, combo in enumerate(combos):
        cy = fig_h - tmargin - (r_idx + 0.5) * row_h
        ax.text(lmargin - 0.1, cy, combo,
                ha="right", va="center", fontsize=8, color="#111", zorder=3)

        for c_idx, model in enumerate(models):
            cx = lmargin + (c_idx + 0.5) * col_w

            for val, x_off in [(v10[r_idx, c_idx], -sub_w / 2),
                                (v2[r_idx,  c_idx], +sub_w / 2)]:
                if np.isnan(val):
                    ax.text(cx + x_off, cy, "—", ha="center", va="center",
                            fontsize=7.5, color="#aaa", zorder=4)
                    continue
                is_best = abs(val - row_max[r_idx, 0]) < 0.001
                if is_best:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (cx + x_off - sub_w / 2 + 0.03,
                         fig_h - tmargin - (r_idx + 1) * row_h + 0.04),
                        sub_w - 0.06, row_h - 0.08,
                        boxstyle="round,pad=0.02", linewidth=1.2,
                        edgecolor="#c0392b", facecolor="#fdecea", zorder=2))
                ax.text(cx + x_off, cy, f"{val:.1f}",
                        ha="center", va="center", fontsize=7.8,
                        fontweight="bold" if is_best else "normal",
                        color="#c0392b" if is_best else "#222", zorder=4)

    # ── Footer ────────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, 0.12,
            "10s = original 10-second epochs  |  2s = pseudo-online 2-second sliding windows  "
            "|  Best per row highlighted in red  |  — = pending HPC results",
            ha="center", va="center", fontsize=6.8, color="#777", zorder=3)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6, color="#111")
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    metric = "F1_macro"

    s10 = load_pivot(os.path.join(RESULTS_DIR, "results_loso.csv"),    metric, ALL_ORDER)
    s2  = load_pivot(os.path.join(RESULTS_DIR, "results_loso_2s.csv"), metric, ALL_ORDER)

    if s10 is None:
        print("Missing results_loso.csv"); return
    if s2 is None:
        print("Missing results_loso_2s.csv"); return

    plot_table(
        s10, s2, metric,
        title="LOSO CV — F1_macro  |  10s epochs vs 2s sliding windows",
        out_path=os.path.join(RESULTS_DIR, "table_window_comparison.png"),
    )


if __name__ == "__main__":
    main()
