"""
Count the number of 10-second windows per participant, directly from label files.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.config import LABEL_DIR, PARTICIPANT_IDS, THRESHOLD

rows = []
for pid in PARTICIPANT_IDS:
    path = os.path.join(LABEL_DIR, f"{pid}.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    total, low, high = 0, 0, 0
    for col in ["level_0", "level_1", "level_2", "level_3"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        total += len(vals)
        low   += int((vals < THRESHOLD).sum())
        high  += int((vals >= THRESHOLD).sum())
    rows.append({"Participant": pid, "Total": total, "Low": low, "High": high})

summary = pd.DataFrame(rows).set_index("Participant")
summary["Low%"]  = (summary["Low"]  / summary["Total"] * 100).round(1)
summary["High%"] = (summary["High"] / summary["Total"] * 100).round(1)

print(summary.to_string())
print(f"\nTotal windows : {summary['Total'].sum()}")
print(f"Total Low     : {summary['Low'].sum()}  ({summary['Low'].sum()/summary['Total'].sum()*100:.1f}%)")
print(f"Total High    : {summary['High'].sum()}  ({summary['High'].sum()/summary['Total'].sum()*100:.1f}%)")
print(f"Min per subj  : {summary['Total'].min()} ({summary['Total'].idxmin()})")
print(f"Max per subj  : {summary['Total'].max()} ({summary['Total'].idxmax()})")
print(f"Mean per subj : {summary['Total'].mean():.1f}")
