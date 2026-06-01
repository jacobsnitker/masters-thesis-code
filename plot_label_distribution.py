"""
Plot raw Likert label distribution (1-9) with counts and binary threshold overlay.
"""

import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.config import LABEL_DIR, PARTICIPANT_IDS, THRESHOLD

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

raw_counts = {}
for pid in PARTICIPANT_IDS:
    path = os.path.join(LABEL_DIR, f"{pid}.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    for col in ["level_0", "level_1", "level_2", "level_3"]:
        if col not in df.columns:
            continue
        for v in df[col].dropna().values:
            k = int(v)
            raw_counts[k] = raw_counts.get(k, 0) + 1

values = list(range(1, 10))
counts = [raw_counts.get(v, 0) for v in values]
total  = sum(counts)

low_color  = "#5b9bd5"
high_color = "#ed7d31"
colors = [low_color if v < THRESHOLD else high_color for v in values]

fig, ax = plt.subplots(figsize=(7, 4.2))

bars = ax.bar(values, counts, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

# Count + percentage labels on each bar
for bar, count in zip(bars, counts):
    pct = count / total * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 12,
        f"{count}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=8, color="#333", zorder=4
    )

# Threshold divider
ax.axvline(x=THRESHOLD - 0.5, color="#555", linewidth=1.5, linestyle="--", zorder=5)
ax.text(THRESHOLD - 0.55, max(counts) * 0.97, "Threshold",
        ha="right", va="top", fontsize=8, color="#555", style="italic")

# Binary totals
low_total  = sum(raw_counts.get(v, 0) for v in range(1, THRESHOLD))
high_total = sum(raw_counts.get(v, 0) for v in range(THRESHOLD, 10))

ax.set_xlabel("NASA-TLX Cognitive Load Rating", fontsize=10)
ax.set_ylabel("Number of Labels", fontsize=10)
ax.set_title("Raw Label Distribution across All Participants and Sessions", fontsize=11, fontweight="bold")
ax.set_xticks(values)
ax.set_xlim(0.4, 9.6)
ax.set_ylim(0, max(counts) * 1.22)
ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

low_patch  = mpatches.Patch(color=low_color,  label=f"Low (< {THRESHOLD}): {low_total} ({100*low_total/total:.1f}%)")
high_patch = mpatches.Patch(color=high_color, label=f"High (≥ {THRESHOLD}): {high_total} ({100*high_total/total:.1f}%)")
ax.legend(handles=[low_patch, high_patch], fontsize=9, loc="upper left")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "label_distribution.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out}")
