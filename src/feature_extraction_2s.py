"""
2-second sliding window feature extraction with 50% overlap (1s step).

Each 2s sub-window inherits the binary label of the 10s parent epoch it falls within.
Everything else is identical to feature_extraction.py — same features, same raw window
format, same preprocessing. Only the window duration and step size change.

Window sizes per modality (2s):
  ECG : 2 × 512  = 1024 samples
  EDA : 2 × 128  =  256 samples
  EEG : 2 × 256  =  512 samples
  Gaze: 2 × 50   =  100 samples
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    FS_ECG, FS_EDA, FS_EEG, FS_GAZE, SEGMENT_SEC, PARTICIPANT_IDS, THRESHOLD
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

# ── 2s window parameters ──────────────────────────────────────────────────────
WINDOW_SEC = 2.0   # sub-window duration
STEP_SEC   = 1.0   # 50% overlap


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Shift timestamps so they start at 0."""
    if df.empty or "Timestamp" not in df.columns or len(df) == 0:
        return df.copy()
    df = df.copy()
    df["Timestamp"] -= df["Timestamp"].min()
    return df


def _get_signal_window(df: pd.DataFrame, signal_cols: list[str],
                       t_start: float, t_end: float) -> np.ndarray | None:
    if df.empty or "Timestamp" not in df.columns:
        return None
    available = [c for c in signal_cols if c in df.columns]
    if not available:
        return None
    mask = (df["Timestamp"] >= t_start) & (df["Timestamp"] < t_end)
    seg = df.loc[mask, available]
    if seg.empty:
        return None
    return seg.values.astype(float)


def _get_df_window(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    if df.empty or "Timestamp" not in df.columns:
        return df.copy()
    mask = (df["Timestamp"] >= t_start) & (df["Timestamp"] < t_end)
    return df.loc[mask].copy()


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
    seg  = seg.ffill().bfill().fillna(0.0)
    seg  = seg.values.astype(np.float32)
    if len(seg) < n_samples // 4:
        return None
    out = np.zeros((len(available), n_samples), dtype=np.float32)
    n   = min(len(seg), n_samples)
    out[:, :n] = seg[:n].T
    return out


def build_combined_dataset_2s(
    participant_ids: list[str] = None,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """
    2-second sliding window version of build_combined_dataset().

    For each 10s parent epoch (one label), generates sub-windows of WINDOW_SEC
    with STEP_SEC step. Each sub-window inherits the parent epoch's binary label.

    Returns:
      X        – DataFrame of features,  shape (N, F)
      raw_X    – dict {'ecg','eda','eeg','gaze'}, each (N, C, T)
      y        – binary labels (N,)
      subjects – subject index per window (N,)
    """
    if participant_ids is None:
        participant_ids = PARTICIPANT_IDS

    N_ECG  = int(WINDOW_SEC * FS_ECG)    # 1024
    N_EDA  = int(WINDOW_SEC * FS_EDA)    #  256
    N_EEG  = int(WINDOW_SEC * FS_EEG)    #  512
    N_GAZE = int(WINDOW_SEC * FS_GAZE)   #  100

    EEG_COLS      = ["TP9", "AF7", "AF8", "TP10"]
    ECG_COLS_RAW  = ["LL_RA"]
    EDA_COLS_RAW  = ["EDA_filtered"]
    GAZE_COLS_RAW = ["ET_PupilLeft"]

    # Sub-window offsets within a 10s parent epoch
    sub_offsets = np.arange(0, SEGMENT_SEC - WINDOW_SEC + STEP_SEC, STEP_SEC)
    # e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8] → 9 sub-windows per 10s epoch

    all_rows  = []
    ecg_wins, eda_wins, eeg_wins, gaze_wins = [], [], [], []
    all_labels, all_subj = [], []

    for subj_idx, pid in enumerate(tqdm(participant_ids, desc="Participants (2s)")):
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
                # Parent 10s epoch boundaries
                parent_t0 = idx * SEGMENT_SEC
                parent_t1 = parent_t0 + SEGMENT_SEC

                for offset in sub_offsets:
                    t0 = parent_t0 + offset
                    t1 = t0 + WINDOW_SEC

                    # Clamp sub-window to stay within parent epoch
                    if t1 > parent_t1:
                        break

                    row = {}

                    # ── ECG features ───────────────────────────────────────────
                    ecg_win = _get_signal_window(ecg_df, ECG_FEAT_COLS, t0, t1)
                    if ecg_win is not None and ecg_win.shape[0] > 27:
                        ecg_feats = extract_ecg_features(ecg_win[:, 0], fs=FS_ECG)
                    else:
                        ecg_feats = {}
                    row.update({f"ecg_{k}": v for k, v in ecg_feats.items()})

                    # ── EDA features ───────────────────────────────────────────
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

                    # ── EEG features ───────────────────────────────────────────
                    eeg_win_arr = _get_signal_window(eeg_df, EEG_COLS, t0, t1)
                    if eeg_win_arr is not None and eeg_win_arr.shape[0] > 27:
                        ch_dict = {ch: eeg_win_arr[:, i] for i, ch in enumerate(EEG_COLS)}
                        eeg_feats = extract_eeg_features(ch_dict, fs=FS_EEG)
                    else:
                        eeg_feats = {}
                    row.update({f"eeg_{k}": v for k, v in eeg_feats.items()})

                    # ── Gaze features ──────────────────────────────────────────
                    gaze_win = _get_df_window(gaze_df, t0, t1)
                    if not gaze_win.empty:
                        gaze_feats = extract_gaze_features(gaze_win)
                    else:
                        gaze_feats = {}
                    row.update({f"gaze_{k}": v for k, v in gaze_feats.items()})

                    # Skip if no features extracted
                    if not row:
                        continue

                    # ── Raw windows ────────────────────────────────────────────
                    ecg_w  = _extract_raw_window(ecg_df,  ECG_COLS_RAW,  t0, t1, N_ECG)
                    eda_w  = _extract_raw_window(eda_df,  EDA_COLS_RAW,  t0, t1, N_EDA)
                    eeg_w  = _extract_raw_window(eeg_df,  EEG_COLS,      t0, t1, N_EEG)
                    gaze_w = _extract_raw_window(gaze_df, GAZE_COLS_RAW, t0, t1, N_GAZE)

                    all_rows.append(row)
                    ecg_wins.append( ecg_w  if ecg_w  is not None else np.zeros((1, N_ECG),  np.float32))
                    eda_wins.append( eda_w  if eda_w  is not None else np.zeros((1, N_EDA),  np.float32))
                    eeg_wins.append( eeg_w  if eeg_w  is not None else np.zeros((4, N_EEG),  np.float32))
                    gaze_wins.append(gaze_w if gaze_w is not None else np.zeros((1, N_GAZE), np.float32))
                    all_labels.append(int(label))   # inherit parent epoch label
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
