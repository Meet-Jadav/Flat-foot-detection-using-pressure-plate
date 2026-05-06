import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

DF_PATH = 'phase2_ml_models/dataset.csv'

if __name__ == '__main__':
    df = pd.read_csv(DF_PATH)
    y = df['label'].astype(int)
    print('Loaded', len(df), 'rows')
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['label']]
    results = []
    for c in numeric_cols:
        try:
            if len(np.unique(df[c]))<2:
                continue
            auc = roc_auc_score(y, df[c])
        except Exception:
            auc = np.nan
        if not np.isnan(auc) and (auc > 0.99 or auc < 0.01):
            print(f'POTENTIAL LEAKAGE: {c} AUC={auc:.6f}')
        results.append((c, auc))
    # print top features by AUC distance from 0.5
    ranked = sorted(results, key=lambda x: abs(0.5 - (x[1] if not np.isnan(x[1]) else 0.5)), reverse=True)[:15]
    print('\nTop 15 features by AUC distance from 0.5:')
    for c, auc in ranked:
        print(f'  {c}: {auc}')
