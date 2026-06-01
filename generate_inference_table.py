import pandas as pd
import numpy as np
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_ORDER = ['LDA', 'LR', 'MLP', 'GB', 'XGBoost', 'RF', 'SVM', 'LGBM']

df = pd.read_csv(os.path.join(RESULTS_DIR, "inference_time_combined.csv"))
summary = df.groupby(['model', 'window'])['ms_per_sample'].agg(['mean', 'std']).reset_index()
p10 = summary[summary.window == '10s'].set_index('model')
p2s = summary[summary.window == '2s'].set_index('model')

lines = [
    r'\begin{table}[!htbp]',
    r'  \centering',
    r'  \caption{Mean per-sample inference time (ms) averaged across all 15 modality '
    r'combinations, for 10s and 2s windows. Measured on an 80/20 stratified hold-out set '
    r'(median of 10 repeats). CNN and Transformer omitted (raw-signal inputs).}',
    r'  \label{tab:inference_time}',
    r'  \begin{tabular}{lrr}',
    r'    \toprule',
    r'    \textbf{Model} & \textbf{10s (ms/sample)} & \textbf{2s (ms/sample)} \\',
    r'    \midrule',
]

for name in MODEL_ORDER:
    m10, s10 = p10.loc[name, 'mean'], p10.loc[name, 'std']
    m2,  s2  = p2s.loc[name, 'mean'], p2s.loc[name, 'std']
    lines.append(f'    {name} & ${m10:.4f} \\pm {s10:.4f}$ & ${m2:.4f} \\pm {s2:.4f}$ \\\\')

lines += [
    r'    \bottomrule',
    r'  \end{tabular}',
    r'\end{table}',
]

tex = '\n'.join(lines)
out = os.path.join(RESULTS_DIR, "table_inference_time.tex")
with open(out, 'w') as f:
    f.write(tex + '\n')
print(tex)
print(f'\nSaved -> {out}')
