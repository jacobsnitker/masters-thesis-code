"""
Gaze feature extraction from a 10-second window (paper Table II).

Features:
  Min/Max/Mean of: left pupil, right pupil, blink, fixation duration,
                   fixation dispersion, saccade duration, saccade amplitude,
                   saccade peak velocity, saccade peak acceleration,
                   saccade peak deceleration, saccade direction
  Count of: blinks, fixations, saccades
"""

import numpy as np
import pandas as pd


def _col_stats(arr: np.ndarray, prefix: str) -> dict:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {f"{prefix}_min": 0.0, f"{prefix}_max": 0.0, f"{prefix}_mean": 0.0}
    return {
        f"{prefix}_min":  float(np.min(valid)),
        f"{prefix}_max":  float(np.max(valid)),
        f"{prefix}_mean": float(np.mean(valid)),
    }


# Columns → feature prefix mapping
_STAT_COLS = {
    "ET_PupilLeft":              "pupil_left",
    "ET_PupilRight":             "pupil_right",
    "Blink detected (binary)":   "blink",
    "Fixation Duration":         "fixation_duration",
    "Fixation Dispersion":       "fixation_dispersion",
    "Saccade Duration":          "saccade_duration",
    "Saccade Amplitude":         "saccade_amplitude",
    "Saccade Peak Velocity":     "saccade_peak_velocity",
    "Saccade Peak Acceleration": "saccade_peak_accel",
    "Saccade Peak Deceleration": "saccade_peak_decel",
    "Saccade Direction":         "saccade_direction",
}


def extract_gaze_features(window_df: pd.DataFrame) -> dict:
    feats = {}

    for col, prefix in _STAT_COLS.items():
        if col in window_df.columns:
            arr = window_df[col].values.astype(float)
        else:
            arr = np.array([np.nan])
        feats.update(_col_stats(arr, prefix))

    # Count of blinks
    if "Blink detected (binary)" in window_df.columns:
        blink_arr = window_df["Blink detected (binary)"].fillna(0).values.astype(float)
        feats["n_blinks"] = float(np.sum(blink_arr > 0))
    else:
        feats["n_blinks"] = 0.0

    # Count of fixations (non-NaN fixation duration rows)
    if "Fixation Duration" in window_df.columns:
        feats["n_fixations"] = float(window_df["Fixation Duration"].notna().sum())
    else:
        feats["n_fixations"] = 0.0

    # Count of saccades (non-NaN saccade duration rows)
    if "Saccade Duration" in window_df.columns:
        feats["n_saccades"] = float(window_df["Saccade Duration"].notna().sum())
    else:
        feats["n_saccades"] = 0.0

    return feats
