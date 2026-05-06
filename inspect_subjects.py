import pandas as pd

df=pd.read_csv('phase2_ml_models/dataset.csv')
print('total samples', len(df))
print('subject_id unique count:', df['subject_id'].nunique())
print(df['subject_id'].value_counts().head(20))
print('\nlabels per subject (unique counts):')
print(df.groupby('subject_id')['label'].nunique().sort_values().head(20))
