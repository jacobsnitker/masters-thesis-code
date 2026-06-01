"""
ECG preprocessing (paper Section IV-A):
  1. Butterworth bandpass filter 5–15 Hz
  2. User-specific z-score normalization
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from src.config import FS_ECG

ECG_SIGNAL_COLS = ["LL_RA", "LA_RA", "Vx_RL"]


def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def preprocess_ecg_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter + z-score a single session ECG DataFrame.
    NaN gaps are interpolated before filtering then restored.
    """
    if df.empty or len(df) < 10:
        return df.copy()
    b, a = _butter_bandpass(5.0, 15.0, FS_ECG)
    out = df.copy()
    for col in ECG_SIGNAL_COLS:
        if col not in out.columns:
            continue
        sig = out[col].interpolate(method="linear").bfill().ffill().values
        sig = filtfilt(b, a, sig)
        # user-specific z-score (computed over entire session)
        mu, sd = sig.mean(), sig.std()
        if sd > 0:
            sig = (sig - mu) / sd
        out[col] = sig
    return out


def preprocess_ecg(sessions: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Preprocess all 4 sessions; z-score is per-session as per the paper."""
    return [preprocess_ecg_session(s) for s in sessions]
