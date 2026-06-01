"""
Generate LaTeX tables for 10-fold and LOSO results.
Each cell shows Accuracy (F1_macro), matching the CLARE paper table style.
Best F1 per row is bolded. Uses table* (full-width) with scriptsize font.
Requires: \\usepackage{booktabs} in preamble.
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

COMBO_ORDER = [
    "ECG", "EDA", "EEG", "Gaze",
    "ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze",
    "ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze", "All",
]

MODEL_ORDER = ["GB", "LGBM", "LDA", "LR", "MLP", "RF", "SVM", "XGBoost", "CNN", "Transformer"]

GROUPS = [
    ("Single",    ["ECG", "EDA", "EEG", "Gaze"]),
    ("Bi-modal",  ["ECG+EDA", "ECG+EEG", "ECG+Gaze", "EDA+EEG", "EDA+Gaze", "EEG+Gaze"]),
    ("Tri-modal", ["ECG+EDA+EEG", "ECG+EDA+Gaze", "ECG+EEG+Gaze", "EDA+EEG+Gaze"]),
    ("All",       ["All"]),
]


def load_pivots(path: str):
    df = pd.read_csv(path, header=0, index_col=[0, 1])
    acc = df["Accuracy"].unstack(level=1).reindex(index=COMBO_ORDER, columns=MODEL_ORDER)
    f1  = df["F1_weighted"].unstack(level=1).reindex(index=COMBO_ORDER, columns=MODEL_ORDER)
    return acc, f1


def combo_label(name: str) -> str:
    # "ECG+EDA" → "ECG, EDA" to match paper style
    return name.replace("+", ", ")


def make_latex(acc: pd.DataFrame, f1: pd.DataFrame, caption: str, label: str) -> str:
    n_models = len(MODEL_ORDER)
    col_spec = "l" + "c" * n_models

    header_cols = " & ".join(MODEL_ORDER)

    lines = []
    lines.append(r"\begin{table*}[!htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(rf"  \caption{{{caption}}}")
    lines.append(rf"  \label{{{label}}}")
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Modalities} & "
        r"\multicolumn{" + str(n_models) + r"}{c}{\textbf{Models}} \\"
    )
    lines.append(r"    \cmidrule(l){2-" + str(n_models + 1) + "}")
    lines.append(f"    & {header_cols} \\\\")
    lines.append(r"    \midrule")

    for g_idx, (group_name, combos) in enumerate(GROUPS):
        for combo in combos:
            if combo not in f1.index:
                continue
            f1_row  = f1.loc[combo]
            acc_row = acc.loc[combo]
            best_f1 = f1_row.max()

            cells = []
            for model in MODEL_ORDER:
                a = acc_row[model]
                f = f1_row[model]
                if pd.isna(a) or pd.isna(f):
                    cells.append("—")
                else:
                    s = rf"{a:.2f} ({f:.2f})"
                    if abs(f - best_f1) < 0.05:
                        s = rf"\textbf{{{s}}}"
                    cells.append(s)

            row_str = combo_label(combo) + " & " + " & ".join(cells) + r" \\"
            lines.append("    " + row_str)

        if g_idx < len(GROUPS) - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def save_and_print(tex: str, path: str, title: str):
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(f"Saved → {path}")
    print(f"\n=== {title} ===")
    print(tex)


def main():
    datasets = [
        ("10s", "results_10fold.csv",    "results_loso.csv"),
        ("2s",  "results_10fold_2s.csv", "results_loso_2s.csv"),
    ]

    for tag, file_10, file_lo in datasets:
        acc_10, f1_10 = load_pivots(os.path.join(RESULTS_DIR, file_10))
        acc_lo, f1_lo = load_pivots(os.path.join(RESULTS_DIR, file_lo))

        tex_10fold = make_latex(
            acc_10, f1_10,
            caption=(
                rf"Classifier performance for binary classification of high and low cognitive load "
                rf"using 10-fold cross-validation ({tag} windows). "
                r"Numbers are in Accuracy (F1\textsubscript{weighted}) format. Best F1 per row is \textbf{bold}."
            ),
            label=f"tab:results_10fold_{tag.replace('s', 's')}",
        )

        tex_loso = make_latex(
            acc_lo, f1_lo,
            caption=(
                rf"Classifier performance for binary classification of high and low cognitive load "
                rf"using leave-one-subject-out evaluation ({tag} windows). "
                r"Numbers are in Accuracy (F1\textsubscript{weighted}) format. Best F1 per row is \textbf{bold}."
            ),
            label=f"tab:results_loso_{tag.replace('s', 's')}",
        )

        save_and_print(tex_10fold, os.path.join(RESULTS_DIR, f"table_10fold_{tag}.tex"), f"10-FOLD {tag}")
        print()
        save_and_print(tex_loso,   os.path.join(RESULTS_DIR, f"table_loso_{tag}.tex"),   f"LOSO {tag}")


if __name__ == "__main__":
    main()
