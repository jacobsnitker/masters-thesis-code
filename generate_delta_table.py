"""
Generate LaTeX delta tables: weighted F1 change from 10s to 2s windows (2s minus 10s).
Positive = 2s better, negative = 10s better.
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

GOOD_MODELS = ['GB', 'LGBM', 'LDA', 'LR', 'MLP', 'SVM', 'Transformer']
COMBO_ORDER = [
    'ECG', 'EDA', 'EEG', 'Gaze',
    'ECG+EDA', 'ECG+EEG', 'ECG+Gaze', 'EDA+EEG', 'EDA+Gaze', 'EEG+Gaze',
    'ECG+EDA+EEG', 'ECG+EDA+Gaze', 'ECG+EEG+Gaze', 'EDA+EEG+Gaze', 'All',
]
GROUPS = [
    ('Single',    ['ECG', 'EDA', 'EEG', 'Gaze']),
    ('Bi-modal',  ['ECG+EDA', 'ECG+EEG', 'ECG+Gaze', 'EDA+EEG', 'EDA+Gaze', 'EEG+Gaze']),
    ('Tri-modal', ['ECG+EDA+EEG', 'ECG+EDA+Gaze', 'ECG+EEG+Gaze', 'EDA+EEG+Gaze']),
    ('All',       ['All']),
]


def load(path):
    df = pd.read_csv(path, header=0, index_col=[0, 1])
    return df['F1_weighted'].unstack(level=1).reindex(index=COMBO_ORDER, columns=GOOD_MODELS)


def combo_label(name):
    return name.replace('+', ', ')


def fmt_cell(v):
    if pd.isna(v):
        return '—'
    sign = '+' if v >= 0 else ''
    s = f'{sign}{v:.1f}'
    if v > 1.5:
        return rf'\textbf{{{s}}}'
    if v < -1.5:
        return rf'\textit{{{s}}}'
    return s


def make_latex(delta: pd.DataFrame, caption: str, label: str) -> str:
    n = len(GOOD_MODELS)
    col_spec = 'l' + 'r' * n + 'r'
    header = ' & '.join(GOOD_MODELS) + r' & \textbf{Mean}'

    lines = []
    lines.append(r'\begin{table*}[!htbp]')
    lines.append(r'  \centering')
    lines.append(r'  \scriptsize')
    lines.append(r'  \setlength{\tabcolsep}{3pt}')
    lines.append(rf'  \caption{{{caption}}}')
    lines.append(rf'  \label{{{label}}}')
    lines.append(r'  \begin{tabular}{' + col_spec + '}')
    lines.append(r'    \toprule')
    lines.append(rf'    \textbf{{Modalities}} & {header} \\')
    lines.append(r'    \midrule')

    for g_idx, (group_name, combos) in enumerate(GROUPS):
        for combo in combos:
            if combo not in delta.index:
                continue
            row = delta.loc[combo]
            row_mean = row.mean()
            cells = [fmt_cell(row[m]) for m in GOOD_MODELS]
            cells.append(fmt_cell(row_mean))
            lines.append(f'    {combo_label(combo)} & ' + ' & '.join(cells) + r' \\')
        if g_idx < len(GROUPS) - 1:
            lines.append(r'    \midrule')

    # Column means row
    lines.append(r'    \midrule')
    col_means = [fmt_cell(delta[m].mean()) for m in GOOD_MODELS]
    col_means.append(fmt_cell(delta.mean().mean()))
    lines.append(r'    \textbf{Mean} & ' + ' & '.join(col_means) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


def main():
    p10_10s = load(os.path.join(RESULTS_DIR, 'results_10fold.csv'))
    plo_10s = load(os.path.join(RESULTS_DIR, 'results_loso.csv'))
    p10_2s  = load(os.path.join(RESULTS_DIR, 'results_10fold_2s.csv'))
    plo_2s  = load(os.path.join(RESULTS_DIR, 'results_loso_2s.csv'))

    delta_10fold = (p10_2s - p10_10s).round(1)
    delta_loso   = (plo_2s - plo_10s).round(1)

    tex_10fold = make_latex(
        delta_10fold,
        caption=r'Change in weighted F1 (\%) when using 2s windows instead of 10s windows under 10-fold CV (2s $-$ 10s). '
                r'\textbf{Bold} = gain $>1.5$\,pp; \textit{italic} = loss $>1.5$\,pp.',
        label='tab:delta_10fold',
    )

    tex_loso = make_latex(
        delta_loso,
        caption=r'Change in weighted F1 (\%) when using 2s windows instead of 10s windows under LOSO CV (2s $-$ 10s). '
                r'\textbf{Bold} = gain $>1.5$\,pp; \textit{italic} = loss $>1.5$\,pp.',
        label='tab:delta_loso',
    )

    for tex, fname, title in [
        (tex_10fold, 'table_delta_10fold.tex', '10-FOLD DELTA'),
        (tex_loso,   'table_delta_loso.tex',   'LOSO DELTA'),
    ]:
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, 'w') as f:
            f.write(tex + '\n')
        print(f'Saved → {path}')
        print(f'\n=== {title} ===')
        print(tex)
        print()


if __name__ == '__main__':
    main()
