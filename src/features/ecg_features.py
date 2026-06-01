"""
ECG feature extraction from a 10-second window (paper Table II).

Time domain features:
  Mean/SD/Min/Max of HR, HRV, raw ECG
  RMS HRV, pNN50, pNN20, IQR, MAD, Skewness, Kurtosis, Entropy, AUC², Median

Frequency domain (Welch PSD on RR intervals):
  Peak frequency, absolute power, normalized power for ULF/VLF/LF/HF bands
  LF/HF ratio, Total power
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import neurokit2 as nk
from src.config import FS_ECG, HRV_BANDS


def _safe_entropy(x: np.ndarray, bins: int = 10) -> float:
    counts, _ = np.histogram(x, bins=bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def _auc_squared(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _pnn(rr_ms: np.ndarray, threshold_ms: float) -> float:
    if len(rr_ms) < 2:
        return 0.0
    diffs = np.abs(np.diff(rr_ms))
    return float(np.sum(diffs > threshold_ms) / len(diffs))


def _welch_band_power(rr_interp: np.ndarray, fs_rr: float, fmin: float, fmax: float):
    """Compute absolute power in a frequency band using Welch's method."""
    nperseg = min(len(rr_interp), 256)
    freqs, psd = welch(rr_interp, fs=fs_rr, nperseg=nperseg)
    idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
    if len(idx) == 0:
        return 0.0, 0.0
    abs_power = float(np.trapz(psd[idx], freqs[idx]))
    peak_freq = float(freqs[idx[np.argmax(psd[idx])]])
    return abs_power, peak_freq


def extract_ecg_features(window: np.ndarray, fs: int = FS_ECG) -> dict:
    """
    window : 1-D array of preprocessed ECG (single channel: LL_RA).
    Returns a flat dict of scalar features.
    """
    feats = {}

    # ── R-peak detection → RR intervals ───────────────────────────────────────
    try:
        signals, info = nk.ecg_process(window, sampling_rate=fs)
        rpeaks = info["ECG_R_Peaks"]
        if len(rpeaks) < 2:
            raise ValueError("too few R-peaks")
        rr_samples = np.diff(rpeaks)
        rr_ms      = (rr_samples / fs) * 1000.0   # milliseconds
        hr_bpm     = 60_000.0 / rr_ms              # HR from each RR interval
    except Exception:
        # Fallback: return zeros if R-peak detection fails
        n_feats = 46  # approximate; caller should handle gracefully
        return {f"ecg_feat_{i}": 0.0 for i in range(n_feats)}

    ecg_raw = window  # z-scored ECG signal

    # ── Time domain ───────────────────────────────────────────────────────────
    for name, arr in [("HR", hr_bpm), ("HRV", rr_ms), ("ECG", ecg_raw)]:
        feats[f"{name}_mean"]     = float(np.mean(arr))
        feats[f"{name}_sd"]       = float(np.std(arr))
        feats[f"{name}_min"]      = float(np.min(arr))
        feats[f"{name}_max"]      = float(np.max(arr))
        feats[f"{name}_median"]   = float(np.median(arr))

    feats["HRV_rms"]    = float(np.sqrt(np.mean(rr_ms ** 2)))
    feats["HRV_pNN50"]  = _pnn(rr_ms, 50.0)
    feats["HRV_pNN20"]  = _pnn(rr_ms, 20.0)

    for name, arr in [("HRV", rr_ms), ("ECG", ecg_raw)]:
        feats[f"{name}_iqr"]      = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        feats[f"{name}_mad"]      = float(np.median(np.abs(arr - np.median(arr))))
        feats[f"{name}_skewness"] = float(skew(arr))
        feats[f"{name}_kurtosis"] = float(kurtosis(arr))
        feats[f"{name}_entropy"]  = _safe_entropy(arr)
        feats[f"{name}_auc2"]     = _auc_squared(arr)

    # ── Frequency domain (Welch on RR series) ─────────────────────────────────
    # Interpolate RR intervals to uniform grid (4 Hz is standard)
    fs_rr = 4.0
    if len(rr_ms) >= 2:
        t_rr = np.cumsum(rr_ms) / 1000.0          # seconds
        t_uni = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_rr)
        if len(t_uni) > 1:
            rr_interp = np.interp(t_uni, t_rr, rr_ms)
        else:
            rr_interp = rr_ms
    else:
        rr_interp = rr_ms

    total_power = 0.0
    band_abs = {}
    band_peak = {}
    for band, (fmin, fmax) in HRV_BANDS.items():
        abs_p, peak_f = _welch_band_power(rr_interp, fs_rr, fmin, fmax)
        band_abs[band]  = abs_p
        band_peak[band] = peak_f
        total_power    += abs_p
        feats[f"HRV_abs_power_{band}"]  = abs_p
        feats[f"HRV_peak_freq_{band}"]  = peak_f

    feats["HRV_total_power"] = total_power
    lf = band_abs.get("LF", 0.0)
    hf = band_abs.get("HF", 0.0)
    feats["HRV_norm_LF"] = lf / total_power if total_power > 0 else 0.0
    feats["HRV_norm_HF"] = hf / total_power if total_power > 0 else 0.0
    feats["HRV_LF_HF"]   = lf / hf if hf > 0 else 0.0

    return feats
