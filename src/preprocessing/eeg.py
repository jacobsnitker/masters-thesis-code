"""
EEG preprocessing (paper Section IV-A):
  1. Butterworth bandpass filter 0.4–90 Hz
     (paper states 0.4–128 Hz; 90 Hz used as practical upper bound < Nyquist)
  2. Notch filter at 60 Hz (Q = 30) to remove powerline noise
  3. User-specific z-score normalization per channel
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from src.config import FS_EEG, EEG_CHANNELS


def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def _notch_filter(freq: float, fs: int, Q: float = 30.0):
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return b, a


def preprocess_eeg_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 10:
        return df.copy()
    b_bp, a_bp = _butter_bandpass(0.4, 90.0, FS_EEG)
    b_n,  a_n  = _notch_filter(60.0, FS_EEG, Q=30.0)

    out = df.copy()
    for ch in EEG_CHANNELS:
        if ch not in out.columns:
            continue
        sig = out[ch].interpolate(method="linear").bfill().ffill().values
        sig = filtfilt(b_bp, a_bp, sig)
        sig = filtfilt(b_n,  a_n,  sig)
        mu, sd = sig.mean(), sig.std()
        if sd > 0:
            sig = (sig - mu) / sd
        out[ch] = sig
    return out


def preprocess_eeg(sessions: list[pd.DataFrame]) -> list[pd.DataFrame]:
    return [preprocess_eeg_session(s) for s in sessions]
