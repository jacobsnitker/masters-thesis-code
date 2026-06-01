"""
Sanity check: count raw label values directly from the label CSV files,
before any binarization, to verify the class distribution.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.config import LABEL_DIR, PARTICIPANT_IDS, THRESHOLD

raw_counts = {}   # raw Likert value (1-9) → count
low, high, total = 0, 0, 0

for pid in PARTICIPANT_IDS:
    path = os.path.join(LABEL_DIR, f"{pid}.csv")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    df = pd.read_csv(path)
    for col in ["level_0", "level_1", "level_2", "level_3"]:
        if col not in df.columns:
            continue
        values = df[col].dropna().values
        for v in values:
            raw_counts[int(v)] = raw_counts.get(int(v), 0) + 1
            if v < THRESHOLD:
                low += 1
            else:
                high += 1
            total += 1

print(f"\nRaw Likert value distribution (1–9):")
for v in sorted(raw_counts):
    bar = "█" * (raw_counts[v] // 5)
    print(f"  {v}: {raw_counts[v]:4d}  {bar}")

print(f"\nBinary distribution (threshold = {THRESHOLD}):")
print(f"  Low  (<  {THRESHOLD}): {low:4d}  ({100*low/total:.1f}%)")
print(f"  High (>= {THRESHOLD}): {high:4d}  ({100*high/total:.1f}%)")
print(f"  Total:       {total:4d}")
