"""
Load raw signals and labels for all participants and sessions.
Returns dicts keyed by participant_id → list of 4 session DataFrames.
"""

import os
import numpy as np
import pandas as pd
from src.config import (
    ECG_DIR, EDA_DIR, EEG_DIR, GAZE_DIR, LABEL_DIR,
    PARTICIPANT_IDS, N_SESSIONS, THRESHOLD,
)


_EMPTY_DF = pd.DataFrame(columns=["Timestamp"])


# ── ECG ────────────────────────────────────────────────────────────────────────
def load_ecg(participant_id: str) -> list[pd.DataFrame]:
    """Return list of N DataFrames (one per available session)."""
    sessions = []
    folder = os.path.join(ECG_DIR, participant_id)
    for s in range(N_SESSIONS):
        path = os.path.join(folder, f"ecg_data_experiment_{s}.csv")
        if not os.path.exists(path):
            sessions.append(_EMPTY_DF.copy())
            continue
        try:
            df = pd.read_csv(path)
            cal_cols = [c for c in df.columns if "CAL" in c]
            df = df[["Timestamp"] + cal_cols].dropna(subset=cal_cols, how="all")
            df = df.rename(columns={
                "ECG LL-RA CAL": "LL_RA",
                "ECG LA-RA CAL": "LA_RA",
                "ECG Vx-RL CAL": "Vx_RL",
            })
            df = df.sort_values("Timestamp").reset_index(drop=True)
            sessions.append(df)
        except Exception:
            sessions.append(_EMPTY_DF.copy())
    return sessions


# ── EDA ────────────────────────────────────────────────────────────────────────
def load_eda(participant_id: str) -> list[pd.DataFrame]:
    """Return list of N DataFrames with columns [Timestamp, EDA]."""
    sessions = []
    folder = os.path.join(EDA_DIR, participant_id)
    for s in range(N_SESSIONS):
        path = os.path.join(folder, f"eda_data_experiment_{s}.csv")
        if not os.path.exists(path):
            sessions.append(_EMPTY_DF.copy())
            continue
        try:
            df = pd.read_csv(path)
            cond_col = "GSR Conductance CAL"
            df = df[["Timestamp", cond_col]].dropna(subset=[cond_col])
            df = df.rename(columns={cond_col: "EDA"})
            df = df.sort_values("Timestamp").reset_index(drop=True)
            sessions.append(df)
        except Exception:
            sessions.append(_EMPTY_DF.copy())
    return sessions


# ── EEG ────────────────────────────────────────────────────────────────────────
def load_eeg(participant_id: str) -> list[pd.DataFrame]:
    """Return list of N DataFrames with columns [Timestamp, TP9, AF7, AF8, TP10]."""
    sessions = []
    folder = os.path.join(EEG_DIR, participant_id)
    for s in range(N_SESSIONS):
        path = os.path.join(folder, f"eeg_data_exp_{s}.csv")
        if not os.path.exists(path):
            sessions.append(_EMPTY_DF.copy())
            continue
        try:
            df = pd.read_csv(path)
            eeg_cols = ["TP9", "AF7", "AF8", "TP10"]
            df = df[["Timestamp"] + eeg_cols].dropna(subset=eeg_cols, how="all")
            df = df.sort_values("Timestamp").reset_index(drop=True)
            sessions.append(df)
        except Exception:
            sessions.append(_EMPTY_DF.copy())
    return sessions


# ── Gaze ───────────────────────────────────────────────────────────────────────
_GAZE_COLS = [
    "Timestamp",
    "ET_PupilLeft", "ET_PupilRight",
    "Blink detected (binary)",
    "Fixation Duration", "Fixation Dispersion",
    "Saccade Duration", "Saccade Amplitude",
    "Saccade Peak Velocity", "Saccade Peak Acceleration",
    "Saccade Peak Deceleration", "Saccade Direction",
]

def load_gaze(participant_id: str) -> list[pd.DataFrame]:
    """Return list of N DataFrames with gaze columns."""
    sessions = []
    folder = os.path.join(GAZE_DIR, participant_id)
    for s in range(N_SESSIONS):
        path = os.path.join(folder, f"gaze_data_experiment_{s}.csv")
        if not os.path.exists(path):
            sessions.append(_EMPTY_DF.copy())
            continue
        try:
            df = pd.read_csv(path)
            available = [c for c in _GAZE_COLS if c in df.columns]
            df = df[available].sort_values("Timestamp").reset_index(drop=True)
            sessions.append(df)
        except Exception:
            sessions.append(_EMPTY_DF.copy())
    return sessions


# ── Labels ─────────────────────────────────────────────────────────────────────
def load_labels(participant_id: str) -> list[np.ndarray]:
    """
    Return list of 4 arrays of binary labels (0=Low, 1=High).
    Each CSV row is one 10-second label; columns are the 4 sessions.
    """
    path = os.path.join(LABEL_DIR, f"{participant_id}.csv")
    df = pd.read_csv(path)
    labels = []
    for col in ["level_0", "level_1", "level_2", "level_3"]:
        if col not in df.columns:
            labels.append(np.array([], dtype=int))
            continue
        raw = df[col].dropna().values
        binary = (raw >= THRESHOLD).astype(int)
        labels.append(binary)
    return labels


# ── Convenience: load everything for one participant ───────────────────────────
def load_participant(participant_id: str) -> dict:
    return {
        "ecg":    load_ecg(participant_id),
        "eda":    load_eda(participant_id),
        "eeg":    load_eeg(participant_id),
        "gaze":   load_gaze(participant_id),
        "labels": load_labels(participant_id),
    }
