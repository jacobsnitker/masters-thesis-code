import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "Data Set", "doi-10.5683-sp3-h0aelt 2")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ECG_DIR   = os.path.join(DATA_DIR, "ECG")
EDA_DIR   = os.path.join(DATA_DIR, "EDA")
EEG_DIR   = os.path.join(DATA_DIR, "EEG")
GAZE_DIR  = os.path.join(DATA_DIR, "Gaze")
LABEL_DIR = os.path.join(DATA_DIR, "Labels")

# ── Auto-extract .rar files if modality directories are missing ────────────────
def _ensure_extracted():
    missing = [d for d in [ECG_DIR, EDA_DIR, EEG_DIR, GAZE_DIR, LABEL_DIR]
               if not os.path.isdir(d)]
    if not missing:
        return
    if not os.path.isdir(DATA_DIR):
        print("ERROR: Dataset not found. Please download the CLARE dataset from:")
        print("  https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT")
        print(f"and place it at: {DATA_DIR}")
        sys.exit(1)
    print("Dataset directories missing — running extraction...")
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "setup_data.py")],
        check=False,
    )
    if result.returncode != 0:
        sys.exit(1)

_ensure_extracted()

# ── Sampling rates ─────────────────────────────────────────────────────────────
FS_ECG  = 512    # Hz
FS_EDA  = 128    # Hz
FS_EEG  = 256    # Hz
FS_GAZE = 50     # Hz

# ── Segment length ─────────────────────────────────────────────────────────────
SEGMENT_SEC = 10  # seconds per label

# ── Label binarization ─────────────────────────────────────────────────────────
# Scores < 5 → Low (0), scores >= 5 → High (1)
THRESHOLD = 5

# ── EEG bands (Hz) ────────────────────────────────────────────────────────────
EEG_BANDS = {
    "delta": (0.4, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 31),
    "gamma": (31, 90),   # upper-bounded at 90 Hz (< Nyquist of 128 Hz)
}
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# ── HRV frequency bands (Hz) ──────────────────────────────────────────────────
HRV_BANDS = {
    "ULF": (0.0,   0.003),
    "VLF": (0.003, 0.04),
    "LF":  (0.04,  0.15),
    "HF":  (0.15,  0.4),
}

# ── Participant IDs ────────────────────────────────────────────────────────────
PARTICIPANT_IDS = sorted([
    d for d in os.listdir(ECG_DIR)
    if os.path.isdir(os.path.join(ECG_DIR, d))
])

N_SESSIONS = 4
