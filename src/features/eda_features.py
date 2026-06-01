"""
EDA feature extraction from a 10-second window (paper Table II).

Features computed on the filtered EDA signal, phasic (SCR), and tonic (SCL):
  Min, Max, Mean, SD, IQR, MAD, Skewness, Kurtosis, Entropy, AUC²
  Phasic-only: amplitude, height, recovery time, rise time, number of peaks
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


def _safe_entropy(x: np.ndarray, bins: int = 10) -> float:
    counts, _ = np.histogram(x, bins=bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _stats(arr: np.ndarray, prefix: str) -> dict:
    if len(arr) == 0:
        arr = np.array([0.0])
    return {
        f"{prefix}_min":      float(np.min(arr)),
        f"{prefix}_max":      float(np.max(arr)),
        f"{prefix}_mean":     float(np.mean(arr)),
        f"{prefix}_sd":       float(np.std(arr)),
        f"{prefix}_iqr":      float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        f"{prefix}_mad":      float(np.median(np.abs(arr - np.median(arr)))),
        f"{prefix}_skewness": float(skew(arr)),
        f"{prefix}_kurtosis": float(kurtosis(arr)),
        f"{prefix}_entropy":  _safe_entropy(arr),
        f"{prefix}_auc2":     float(np.sum(arr ** 2)),
    }


def _phasic_peak_features(phasic: np.ndarray) -> dict:
    """
    Detect SCR peaks and compute amplitude, height (peak value), rise time,
    recovery time, and number of peaks.
    """
    feats = {}
    peaks, props = find_peaks(phasic, height=0, prominence=0.01)
    feats["phasic_n_peaks"] = int(len(peaks))

    if len(peaks) == 0:
        feats["phasic_amplitude_mean"] = 0.0
        feats["phasic_height_mean"]    = 0.0
        feats["phasic_rise_time_mean"] = 0.0
        feats["phasic_recovery_time_mean"] = 0.0
        return feats

    amplitudes = []
    heights    = []
    rise_times = []
    recovery_times = []

    for pk in peaks:
        # amplitude: peak value – preceding trough
        left_base = np.argmin(phasic[:pk]) if pk > 0 else 0
        amplitude = phasic[pk] - phasic[left_base]
        amplitudes.append(amplitude)

        # height: absolute peak value (including tonic)
        heights.append(phasic[pk])

        # rise time: samples from left_base to peak
        rise_times.append(pk - left_base)

        # recovery time: samples from peak back to 50% amplitude
        half = phasic[pk] - 0.5 * amplitude
        right_seg = phasic[pk:]
        cross = np.where(right_seg <= half)[0]
        recovery_times.append(int(cross[0]) if len(cross) > 0 else len(right_seg))

    feats["phasic_amplitude_mean"]     = float(np.mean(amplitudes))
    feats["phasic_height_mean"]        = float(np.mean(heights))
    feats["phasic_rise_time_mean"]     = float(np.mean(rise_times))
    feats["phasic_recovery_time_mean"] = float(np.mean(recovery_times))
    return feats


def extract_eda_features(
    eda_filtered: np.ndarray,
    eda_phasic: np.ndarray,
    eda_tonic: np.ndarray,
) -> dict:
    feats = {}
    feats.update(_stats(eda_filtered, "EDA"))
    feats.update(_stats(eda_phasic,   "EDA_phasic"))
    feats.update(_stats(eda_tonic,    "EDA_tonic"))
    feats.update(_phasic_peak_features(eda_phasic))
    return feats
