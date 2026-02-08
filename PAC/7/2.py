import pandas as pd
import numpy as np

df = pd.read_csv(r'wells_info.csv')
df['CompletionDate'] = pd.to_datetime(df['CompletionDate'])
df['PermitDate'] = pd.to_datetime(df['PermitDate'])
df['month'] = ((df['CompletionDate'] - df['PermitDate']).dt.days/30).round()
print(df['month'])