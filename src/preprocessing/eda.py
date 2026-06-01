"""
EDA preprocessing (paper Section IV-A):
  1. Lowpass Butterworth filter at 3 Hz
  2. User-specific z-score normalization
  3. Highpass filter at 0.05 Hz to separate tonic (SCL) and phasic (SCR)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from src.config import FS_EDA


def _butter_lowpass(cutoff: float, fs: int, order: int = 4):
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return b, a


def _butter_highpass(cutoff: float, fs: int, order: int = 4):
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="high")
    return b, a


def preprocess_eda_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      EDA_filtered  – lowpass-filtered, z-scored signal
      EDA_tonic     – tonic (SCL) component
      EDA_phasic    – phasic (SCR) component
    """
    if df.empty or "EDA" not in df.columns or len(df) < 10:
        out = df.copy()
        out["EDA_filtered"] = out.get("EDA", pd.Series(dtype=float))
        out["EDA_tonic"]    = out.get("EDA", pd.Series(dtype=float))
        out["EDA_phasic"]   = out.get("EDA", pd.Series(dtype=float))
        return out

    b_lp, a_lp = _butter_lowpass(3.0, FS_EDA)
    b_hp, a_hp = _butter_highpass(0.05, FS_EDA)

    out = df.copy()
    sig = out["EDA"].interpolate(method="linear").bfill().ffill().values

    # 1. Lowpass filter
    sig_lp = filtfilt(b_lp, a_lp, sig)

    # 2. Z-score (user-specific, over the full session)
    mu, sd = sig_lp.mean(), sig_lp.std()
    if sd > 0:
        sig_lp = (sig_lp - mu) / sd

    # 3. Tonic = lowpass of the filtered signal (already low-pass)
    #    Phasic = highpass at 0.05 Hz applied to the filtered signal
    tonic  = filtfilt(*_butter_lowpass(0.05, FS_EDA), sig_lp)
    phasic = sig_lp - tonic   # equivalent to highpass

    out["EDA_filtered"] = sig_lp
    out["EDA_tonic"]    = tonic
    out["EDA_phasic"]   = phasic
    return out


def preprocess_eda(sessions: list[pd.DataFrame]) -> list[pd.DataFrame]:
    return [preprocess_eda_session(s) for s in sessions]
