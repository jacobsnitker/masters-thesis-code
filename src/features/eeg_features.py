"""
EEG feature extraction from a 10-second window (paper Table II).

Per channel × per frequency band:
  Frequency domain (Welch PSD): absolute/mean/max/min/median power
  Time domain: spectral entropy, Hjorth mobility & complexity,
               Lempel-Ziv complexity, Higuchi fractal dimension,
               min/max/mean/median of FFT magnitude
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
import antropy as ant
from src.config import FS_EEG, EEG_BANDS, EEG_CHANNELS


def _butter_bandpass_filter(sig: np.ndarray, fmin: float, fmax: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low  = max(fmin, 0.01) / nyq
    high = min(fmax, nyq - 0.5) / nyq
    if low >= high:
        return sig
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def _hjorth(sig: np.ndarray):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    activity   = np.var(sig)
    mobility   = np.sqrt(np.var(d1) / activity) if activity > 0 else 0.0
    complexity = (np.sqrt(np.var(d2) / np.var(d1)) / mobility
                  if (np.var(d1) > 0 and mobility > 0) else 0.0)
    return float(mobility), float(complexity)


def _higuchi_fd(sig: np.ndarray, kmax: int = 10) -> float:
    try:
        return float(ant.higuchi_fd(sig, kmax=kmax))
    except Exception:
        return 0.0


def _lempel_ziv(sig: np.ndarray) -> float:
    try:
        return float(ant.lziv_complexity(sig > np.median(sig), normalize=True))
    except Exception:
        return 0.0


def _spectral_entropy(sig: np.ndarray, fs: int) -> float:
    try:
        return float(ant.spectral_entropy(sig, sf=fs, method="welch", normalize=True))
    except Exception:
        return 0.0


def _band_features(band_sig: np.ndarray, fs: int, prefix: str) -> dict:
    """PSD statistics for one band-filtered signal."""
    feats = {}
    nperseg = min(len(band_sig), 256)
    freqs, psd = welch(band_sig, fs=fs, nperseg=nperseg)
    feats[f"{prefix}_abs_power"]  = float(np.trapz(psd, freqs))
    feats[f"{prefix}_mean_power"] = float(np.mean(psd))
    feats[f"{prefix}_max_power"]  = float(np.max(psd))
    feats[f"{prefix}_min_power"]  = float(np.min(psd))
    feats[f"{prefix}_med_power"]  = float(np.median(psd))
    return feats


def extract_eeg_features(windows: dict[str, np.ndarray], fs: int = FS_EEG) -> dict:
    """
    windows: dict of {channel_name: 1-D numpy array} for one 10-second segment.
    Returns flat feature dict.
    """
    feats = {}

    for ch in EEG_CHANNELS:
        sig = windows.get(ch)
        if sig is None or len(sig) == 0:
            continue

        # ── Time-domain features (whole channel signal) ───────────────────────
        fft_mag = np.abs(np.fft.rfft(sig))
        feats[f"{ch}_fft_min"]    = float(np.min(fft_mag))
        feats[f"{ch}_fft_max"]    = float(np.max(fft_mag))
        feats[f"{ch}_fft_mean"]   = float(np.mean(fft_mag))
        feats[f"{ch}_fft_median"] = float(np.median(fft_mag))

        feats[f"{ch}_spectral_entropy"] = _spectral_entropy(sig, fs)
        mob, comp = _hjorth(sig)
        feats[f"{ch}_hjorth_mobility"]   = mob
        feats[f"{ch}_hjorth_complexity"] = comp
        feats[f"{ch}_lziv_complexity"]   = _lempel_ziv(sig)
        feats[f"{ch}_higuchi_fd"]        = _higuchi_fd(sig)

        # ── Per-band features ─────────────────────────────────────────────────
        for band, (fmin, fmax) in EEG_BANDS.items():
            band_sig = _butter_bandpass_filter(sig, fmin, fmax, fs)
            prefix   = f"{ch}_{band}"
            feats.update(_band_features(band_sig, fs, prefix))

    return feats
