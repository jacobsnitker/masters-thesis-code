"""
Extract the CLARE dataset .rar files.

Run this once after downloading the dataset:
    python setup_data.py

Requires one of the following to be installed:
  - unrar:  brew install unrar          (macOS)
            apt-get install unrar       (Linux)
  - 7z:     brew install p7zip          (macOS)
            apt-get install p7zip-full  (Linux)
  - rarfile Python package: pip install rarfile  (also needs unrar binary)
"""

import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data Set", "doi-10.5683-sp3-h0aelt 2")

MODALITIES = ["ECG", "EDA", "EEG", "Gaze", "Labels"]


def _try_unrar(rar_path, out_dir):
    result = subprocess.run(
        ["unrar", "x", "-o+", rar_path, out_dir],
        capture_output=True,
    )
    return result.returncode == 0


def _try_7z(rar_path, out_dir):
    result = subprocess.run(
        ["7z", "x", rar_path, f"-o{out_dir}", "-y"],
        capture_output=True,
    )
    return result.returncode == 0


def _try_rarfile(rar_path, out_dir):
    try:
        import rarfile
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(out_dir)
        return True
    except Exception:
        return False


def extract(rar_path, out_dir):
    for fn in [_try_unrar, _try_7z, _try_rarfile]:
        try:
            if fn(rar_path, out_dir):
                return True
        except FileNotFoundError:
            continue
    return False


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Dataset directory not found:\n  {DATA_DIR}")
        print("\nPlease download the CLARE dataset from:")
        print("  https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT")
        print("and place it at the path above.")
        sys.exit(1)

    any_extracted = False
    for name in MODALITIES:
        out_dir = os.path.join(DATA_DIR, name)
        rar_path = os.path.join(DATA_DIR, f"{name}.rar")

        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"  {name}: already extracted, skipping.")
            continue

        if not os.path.isfile(rar_path):
            print(f"  {name}: {rar_path} not found — skipping.")
            continue

        print(f"  {name}: extracting {rar_path} ...", end=" ", flush=True)
        if extract(rar_path, DATA_DIR):
            print("done.")
            any_extracted = True
        else:
            print("FAILED.")
            print(f"\nCould not extract {rar_path}.")
            print("Please install unrar or 7z and try again:")
            print("  macOS:  brew install unrar")
            print("  Linux:  sudo apt-get install unrar")
            sys.exit(1)

    if any_extracted:
        print("\nAll modalities extracted successfully.")
    else:
        print("\nAll modalities already extracted — nothing to do.")


if __name__ == "__main__":
    main()
