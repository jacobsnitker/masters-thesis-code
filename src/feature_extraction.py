"""
Orchestrates feature extraction:
  - Segments each preprocessed session into 10-second windows aligned to labels
  - Extracts ECG, EDA, EEG, Gaze features per window
  - Returns (X, y, subject_ids) arrays ready for classification
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    FS_ECG, FS_EDA, FS_EEG, FS_GAZE, SEGMENT_SEC, PARTICIPANT_IDS
)
from src.data_loader import load_participant
from src.preprocessing.ecg  import preprocess_ecg
from src.preprocessing.eda  import preprocess_eda
from src.preprocessing.eeg  import preprocess_eeg
from src.preprocessing.gaze import preprocess_gaze
from src.features.ecg_features  import extract_ecg_features
from src.features.eda_features  import extract_eda_features
from src.features.eeg_features  import extract_eeg_features
from src.features.gaze_features import extract_gaze_features


def _get_signal_window(df: pd.DataFrame, signal_cols: list[str],
                       t_start: float, t_end: float, fs: int) -> np.ndarray | None:
    """Slice a DataFrame by timestamp and return signal array. Returns None if empty."""
    if df.empty or "Timestamp" not in df.columns:
        return None
    available = [c for c in signal_cols if c in df.columns]
    if not available:
        return None
    mask = (df["Timestamp"] >= t_start) & (df["Timestamp"] < t_end)
    seg = df.loc[mask, available]
    if seg.empty:
        return None
    # Resample to expected number of samples
    n_expected = int(SEGMENT_SEC * fs)
    arr = seg.values.astype(float)
    if len(arr) < n_expected // 2:   # less than half the expected → skip
        return None
    return arr


def _get_df_window(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    if df.empty or "Timestamp" not in df.columns:
        return df.copy()
    mask = (df["Timestamp"] >= t_start) & (df["Timestamp"] < t_end)
    return df.loc[mask].copy()


def extract_features_one_session(
    ecg_df:  pd.DataFrame,
    eda_df:  pd.DataFrame,
    eeg_df:  pd.DataFrame,
    gaze_df: pd.DataFrame,
    labels:  np.ndarray,
) -> tuple[list[dict], list[int]]:
    """
    Segment each session into N 10-second windows (one per label).
    The timestamps in each signal DataFrame define the recording duration.
    Labels are indexed 0…N-1 corresponding to seconds 0, 10, 20, …
    """
    ECG_COLS  = [c for c in ["LL_RA", "LA_RA", "Vx_RL"] if c in ecg_df.columns]
    EEG_COLS  = ["TP9", "AF7", "AF8", "TP10"]

    # Normalize timestamps to start at 0 (recording offsets differ across modalities)
    def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "Timestamp" not in df.columns or len(df) == 0:
            return df.copy()
        df = df.copy()
        df["Timestamp"] -= df["Timestamp"].min()
        return df

    ecg_df  = _normalize_ts(ecg_df)
    eda_df  = _normalize_ts(eda_df)
    eeg_df  = _normalize_ts(eeg_df)
    gaze_df = _normalize_ts(gaze_df)

    feature_rows = []
    valid_labels = []

    for idx, label in enumerate(labels):
        t_start = idx * SEGMENT_SEC
        t_end   = t_start + SEGMENT_SEC

        row = {}

        # ── ECG ────────────────────────────────────────────────────────────────
        ecg_win = _get_signal_window(ecg_df, ECG_COLS, t_start, t_end, FS_ECG)
        if ecg_win is not None and ecg_win.shape[0] > 10:
            # Use LL_RA channel (first available) for R-peak detection
            ecg_feats = extract_ecg_features(ecg_win[:, 0], fs=FS_ECG)
        else:
            ecg_feats = {}
        row.update({f"ecg_{k}": v for k, v in ecg_feats.items()})

        # ── EDA ────────────────────────────────────────────────────────────────
        eda_win = _get_df_window(eda_df, t_start, t_end)
        if not eda_win.empty and len(eda_win) > 5:
            eda_feats = extract_eda_features(
                eda_win["EDA_filtered"].values,
                eda_win["EDA_phasic"].values,
                eda_win["EDA_tonic"].values,
            )
        else:
            eda_feats = {}
        row.update({f"eda_{k}": v for k, v in eda_feats.items()})

        # ── EEG ────────────────────────────────────────────────────────────────
        eeg_win_arr = _get_signal_window(eeg_df, EEG_COLS, t_start, t_end, FS_EEG)
        if eeg_win_arr is not None and eeg_win_arr.shape[0] > 10:
            ch_dict = {ch: eeg_win_arr[:, i] for i, ch in enumerate(EEG_COLS)}
            eeg_feats = extract_eeg_features(ch_dict, fs=FS_EEG)
        else:
            eeg_feats = {}
        row.update({f"eeg_{k}": v for k, v in eeg_feats.items()})

        # ── Gaze ───────────────────────────────────────────────────────────────
        gaze_win = _get_df_window(gaze_df, t_start, t_end)
        if not gaze_win.empty:
            gaze_feats = extract_gaze_features(gaze_win)
        else:
            gaze_feats = {}
        row.update({f"gaze_{k}": v for k, v in gaze_feats.items()})

        if row:
            feature_rows.append(row)
            valid_labels.append(int(label))

    return feature_rows, valid_labels


def build_dataset(participant_ids: list[str] = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Process all participants and sessions.
    Returns:
      X           – DataFrame of features, shape (N_windows, N_features)
      y           – binary label array, shape (N_windows,)
      subject_ids – participant index per window, shape (N_windows,) — used for LOSO
    """
    if participant_ids is None:
        participant_ids = PARTICIPANT_IDS

    all_rows    = []
    all_labels  = []
    all_subj    = []

    for subj_idx, pid in enumerate(tqdm(participant_ids, desc="Participants")):
        data = load_participant(pid)

        ecg_sessions  = preprocess_ecg(data["ecg"])
        eda_sessions  = preprocess_eda(data["eda"])
        eeg_sessions  = preprocess_eeg(data["eeg"])
        gaze_sessions = preprocess_gaze(data["gaze"])
        label_sessions = data["labels"]

        for sess in range(len(label_sessions)):
            rows, lbls = extract_features_one_session(
                ecg_sessions[sess],
                eda_sessions[sess],
                eeg_sessions[sess],
                gaze_sessions[sess],
                label_sessions[sess],
            )
            all_rows.extend(rows)
            all_labels.extend(lbls)
            all_subj.extend([subj_idx] * len(lbls))

    X = pd.DataFrame(all_rows).fillna(0.0)
    y = np.array(all_labels, dtype=int)
    subjects = np.array(all_subj, dtype=int)
    return X, y, subjects


# ── Raw signal windows for CNN ────────────────────────────────────────────────
def _extract_raw_window(df: pd.DataFrame, sig_cols: list[str],
                        t_start: float, t_end: float,
                        n_samples: int) -> np.ndarray | None:
    """Return a zero-padded/cropped array of shape (len(sig_cols), n_samples)."""
    if df.empty or "Timestamp" not in df.columns:
        return None
    available = [c for c in sig_cols if c in df.columns]
    if not available:
        return None
    mask = (df["Timestamp"] >= t_start) & (df["Timestamp"] < t_end)
    seg  = df.loc[mask, available].copy()
    # Forward-fill then back-fill NaNs (blinks/missing gaze samples)
    seg  = seg.ffill().bfill().fillna(0.0)
    seg  = seg.values.astype(np.float32)
    if len(seg) < n_samples // 4:   # too sparse → skip
        return None
    # Pad or crop each channel to exactly n_samples
    out = np.zeros((len(available), n_samples), dtype=np.float32)
    n   = min(len(seg), n_samples)
    out[:, :n] = seg[:n].T
    return out  # (C, n_samples)


def build_combined_dataset(
    participant_ids: list[str] = None,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """
    Single-pass build: extracts handcrafted features AND raw windows in one loop,
    guaranteeing perfect index alignment between them.

    Returns:
      X        – DataFrame of features,  shape (N, F)
      raw_X    – dict {'ecg','eda','eeg','gaze'}, each (N, C, T)
      y        – binary labels (N,)
      subjects – subject index per window (N,)
    """
    if participant_ids is None:
        participant_ids = PARTICIPANT_IDS

    N_ECG  = int(SEGMENT_SEC * FS_ECG)    # 5120
    N_EDA  = int(SEGMENT_SEC * FS_EDA)    # 1280
    N_EEG  = int(SEGMENT_SEC * FS_EEG)    # 2560
    N_GAZE = int(SEGMENT_SEC * FS_GAZE)   # 500

    EEG_COLS  = ["TP9", "AF7", "AF8", "TP10"]
    ECG_COLS_RAW  = ["LL_RA"]
    EDA_COLS_RAW  = ["EDA_filtered"]
    GAZE_COLS_RAW = ["ET_PupilLeft"]

    all_rows  = []
    ecg_wins, eda_wins, eeg_wins, gaze_wins = [], [], [], []
    all_labels, all_subj = [], []

    for subj_idx, pid in enumerate(tqdm(participant_ids, desc="Participants")):
        data = load_participant(pid)

        ecg_pp   = preprocess_ecg(data["ecg"])
        eda_pp   = preprocess_eda(data["eda"])
        eeg_pp   = preprocess_eeg(data["eeg"])
        gaze_pp  = preprocess_gaze(data["gaze"])
        labels   = data["labels"]

        for sess in range(len(labels)):
            if len(labels[sess]) == 0:
                continue

            ecg_df  = _normalize_ts(ecg_pp[sess])
            eda_df  = _normalize_ts(eda_pp[sess])
            eeg_df  = _normalize_ts(eeg_pp[sess])
            gaze_df = _normalize_ts(gaze_pp[sess])

            ECG_FEAT_COLS = [c for c in ["LL_RA", "LA_RA", "Vx_RL"] if c in ecg_df.columns]

            for idx, label in enumerate(labels[sess]):
                t0 = idx * SEGMENT_SEC
                t1 = t0 + SEGMENT_SEC

                row = {}

                # ── Features ───────────────────────────────────────────────────
                ecg_win = _get_signal_window(ecg_df, ECG_FEAT_COLS, t0, t1, FS_ECG)
                if ecg_win is not None and ecg_win.shape[0] > 10:
                    ecg_feats = extract_ecg_features(ecg_win[:, 0], fs=FS_ECG)
                else:
                    ecg_feats = {}
                row.update({f"ecg_{k}": v for k, v in ecg_feats.items()})

                eda_win = _get_df_window(eda_df, t0, t1)
                if not eda_win.empty and len(eda_win) > 5:
                    eda_feats = extract_eda_features(
                        eda_win["EDA_filtered"].values,
                        eda_win["EDA_phasic"].values,
                        eda_win["EDA_tonic"].values,
                    )
                else:
                    eda_feats = {}
                row.update({f"eda_{k}": v for k, v in eda_feats.items()})

                eeg_win_arr = _get_signal_window(eeg_df, EEG_COLS, t0, t1, FS_EEG)
                if eeg_win_arr is not None and eeg_win_arr.shape[0] > 10:
                    ch_dict = {ch: eeg_win_arr[:, i] for i, ch in enumerate(EEG_COLS)}
                    eeg_feats = extract_eeg_features(ch_dict, fs=FS_EEG)
                else:
                    eeg_feats = {}
                row.update({f"eeg_{k}": v for k, v in eeg_feats.items()})

                gaze_win = _get_df_window(gaze_df, t0, t1)
                if not gaze_win.empty:
                    gaze_feats = extract_gaze_features(gaze_win)
                else:
                    gaze_feats = {}
                row.update({f"gaze_{k}": v for k, v in gaze_feats.items()})

                # Skip if no features extracted (same criterion as build_dataset)
                if not row:
                    continue

                # ── Raw windows (same window, guaranteed aligned) ───────────────
                ecg_w  = _extract_raw_window(ecg_df,  ECG_COLS_RAW,  t0, t1, N_ECG)
                eda_w  = _extract_raw_window(eda_df,  EDA_COLS_RAW,  t0, t1, N_EDA)
                eeg_w  = _extract_raw_window(eeg_df,  EEG_COLS,      t0, t1, N_EEG)
                gaze_w = _extract_raw_window(gaze_df, GAZE_COLS_RAW, t0, t1, N_GAZE)

                all_rows.append(row)
                ecg_wins.append( ecg_w  if ecg_w  is not None else np.zeros((1, N_ECG),  np.float32))
                eda_wins.append( eda_w  if eda_w  is not None else np.zeros((1, N_EDA),  np.float32))
                eeg_wins.append( eeg_w  if eeg_w  is not None else np.zeros((4, N_EEG),  np.float32))
                gaze_wins.append(gaze_w if gaze_w is not None else np.zeros((1, N_GAZE), np.float32))
                all_labels.append(int(label))
                all_subj.append(subj_idx)

    X = pd.DataFrame(all_rows).fillna(0.0)
    raw_X = {
        "ecg":  np.stack(ecg_wins,  axis=0),
        "eda":  np.stack(eda_wins,  axis=0),
        "eeg":  np.stack(eeg_wins,  axis=0),
        "gaze": np.stack(gaze_wins, axis=0),
    }
    y        = np.array(all_labels, dtype=int)
    subjects = np.array(all_subj,   dtype=int)
    return X, raw_X, y, subjects


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Shift timestamps so they start at 0."""
    if df.empty or "Timestamp" not in df.columns or len(df) == 0:
        return df.copy()
    df = df.copy()
    df["Timestamp"] -= df["Timestamp"].min()
    return df
