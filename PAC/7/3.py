import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  # None - отображать все
pd.set_option('display.width', None)        # Без ограничения ширины
pd.set_option('display.max_colwidth', None) # Без ограничения ширины колонок

df = pd.read_csv(r'wells_info_na.csv')
# print(df.isnull().sum())
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
# print(df['formation'].mode())
print(df)