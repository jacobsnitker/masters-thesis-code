# CLARE Replication Study

Replication and extension of the [CLARE benchmark](https://github.com/Prithila05/CLARE) (Bhatti et al., 2025) for multimodal cognitive load classification. This study evaluates 10 classifiers across 15 modality combinations under two evaluation schemes (10-fold CV and LOSO), and extends the original 10-second epoch pipeline with a pseudo-online 2-second sliding window transformation.

---

## Dataset

The CLARE dataset must be downloaded separately from the [official repository](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT).

After downloading, place the data in the following structure:

```
CLARE REPLICATION STUDY/
└── Data Set/
    └── doi-10.5683-sp3-h0aelt 2/
        ├── ECG/
        ├── EDA/
        ├── EEG/
        ├── Gaze/
        └── Labels/
```

The dataset contains ECG (512 Hz), EDA (128 Hz), EEG (256 Hz), and Gaze (50 Hz) recordings from 20 subjects performing MATB-II tasks, with self-reported cognitive load scores at 10-second intervals.

---

## Setup

### Requirements

- Python 3.11
- macOS (Apple Silicon or Intel) or Linux

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "CLARE REPLICATION STUDY"

# Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
# On Apple Silicon Mac:
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt

# On Intel Mac / Linux:
pip install tensorflow==2.15.0
pip install -r requirements.txt
```

### Data setup

After downloading the dataset and placing it in `Data Set/doi-10.5683-sp3-h0aelt 2/`, the `.rar` files need to be extracted. This happens automatically when you first run `main.py`, but requires `unrar` to be installed:

```bash
# macOS
brew install unar

# Linux
sudo apt-get install unar
```

You can also run the extraction manually:

```bash
python setup_data.py
```

---

## Running the Evaluation

### 10-second epochs (original CLARE pipeline)

```bash
python main.py                              # full pipeline — all modalities, both CV schemes
python main.py --scheme 10fold             # 10-fold CV only
python main.py --scheme loso               # LOSO CV only
python main.py --no-dl                     # skip CNN and Transformer (faster)
python main.py --modalities ECG EDA        # subset of modalities
python main.py --participants 1026 1105    # specific participants (quick test)
python main.py --epochs 10                 # fewer DL epochs (quick test)
```

### 2-second sliding windows (pseudo-online extension)

```bash
python main_2s.py                          # full pipeline
python main_2s.py --scheme 10fold
python main_2s.py --no-dl
python main_2s.py --combo-start 0 --combo-end 5   # run a subset of modality combos
```

Results are saved to `results/results_10fold.csv`, `results/results_loso.csv`, `results/results_10fold_2s.csv`, and `results/results_loso_2s.csv`.

> **Runtime note:** The full pipeline with all 15 modality combinations, 10 classifiers, and both CV schemes takes several hours on a standard laptop. Running with `--no-dl` and a single `--scheme` significantly reduces runtime. The 2-second pipeline is approximately 9× slower than the 10-second pipeline due to the increased number of windows.

---

## Analysis Scripts

Once results CSVs are generated, the following scripts produce tables and figures:

| Script | Output |
|---|---|
| `generate_latex_tables.py` | LaTeX result tables (accuracy + weighted F1) for 10s and 2s |
| `generate_delta_table.py` | LaTeX delta tables showing 2s − 10s F1 change |
| `generate_inference_table.py` | LaTeX inference time table |
| `plot_loss_curves.py` | Training vs validation loss curves (random 80/20 split) |
| `plot_loss_curves_cv.py` | Training vs validation loss curves (proper 10-fold and LOSO splits) |
| `feature_importance.py` | XGBoost gain-based feature importance |
| `feature_importance_all_models.py` | Feature importance for all 8 ML classifiers |
| `measure_inference_time.py` | Per-sample inference time for all classifiers |
| `plot_label_distribution.py` | Label distribution figure |

Run any script from the project root:

```bash
python generate_latex_tables.py
python plot_loss_curves_cv.py   # note: slow — runs full CV for each model
```

---

## Project Structure

```
Clare REPLICATION STUDY/
├── main.py                        # 10-second epoch evaluation entry point
├── main_2s.py                     # 2-second sliding window evaluation entry point
├── requirements.txt
├── src/
│   ├── config.py                  # paths, sampling rates, parameters
│   ├── data_loader.py             # raw signal loading
│   ├── evaluation.py              # 10-fold and LOSO evaluation loops
│   ├── feature_extraction.py      # 10s feature extraction + caching
│   ├── feature_extraction_2s.py   # 2s sliding window feature extraction + caching
│   ├── features/
│   │   ├── ecg_features.py
│   │   ├── eda_features.py
│   │   ├── eeg_features.py
│   │   └── gaze_features.py
│   ├── models/
│   │   ├── ml_classifiers.py      # 8 ML classifiers with CLARE paper hyperparameters
│   │   ├── cnn_model.py           # VGG-style 1D CNN (TensorFlow/Keras)
│   │   └── transformer_model.py   # Transformer network (TensorFlow/Keras)
│   └── preprocessing/
│       ├── ecg.py
│       ├── eda.py
│       ├── eeg.py
│       └── gaze.py
├── results/                       # generated CSVs, figures, and LaTeX tables
└── Data Set/                      # CLARE dataset (not included — download separately)
```

---

## Classifiers

Eight ML classifiers and two deep learning models are evaluated, all using hyperparameters from the CLARE paper:

| Model | Key parameters |
|---|---|
| Gradient Boosting (GB) | n_estimators=300, max_depth=3, loss=log_loss |
| LightGBM (LGBM) | n_estimators=2000, num_leaves=100, lr=0.001 |
| Linear Discriminant Analysis (LDA) | solver=lsqr |
| Logistic Regression (LR) | C=1, max_iter=400 |
| MLP | hidden=(100,10), lr=adaptive, max_iter=1000 |
| Random Forest (RF) | n_estimators=1000, max_depth=5, min_samples_split=5 |
| SVM | C=10 |
| XGBoost | n_estimators=300, lr=0.001, reg_alpha=0.0001 |
| CNN | VGG-style 1D CNN, focal loss (α=4, γ=2), AdaDelta, 100 epochs |
| Transformer | 4-block self-attention, BCE loss, Adam (lr=0.0001), 100 epochs |

---

## Evaluation Schemes

- **10-fold CV:** `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` applied across all windows from all subjects. Note that this allows subject leakage (the same subject's windows appear in both train and test folds), which inflates performance relative to LOSO.
- **LOSO:** `LeaveOneGroupOut` on subject IDs — each fold holds out one subject entirely. This is the more realistic evaluation for generalisation to new users.

---

## Citation

If you use this code, please also cite the original CLARE paper:

```
Bhatti, A., Angkan, P., Behinaein, B., Mahmud, Z., Rodenburg, D., Braund, H., ...
& Hungler, P. (2025). CLARE: Cognitive Load Assessment in Real-time with
Multimodal Data. arXiv:2404.17098.
```
