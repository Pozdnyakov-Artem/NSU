import pandas as pd
import numpy as np

def find_upper_bound(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    upper_bound = q3 + 1.5*iqr

    return upper_bound

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'cinema_sessions_fixed.csv')
df2 = pd.read_csv(r'titanic_with_labels_fixed.csv')

df2 = df2[(df2['sex'].str.lower() == 'м') | (df2['sex'].str.lower() == 'ж')]
df2['sex'] = df2['sex'].map(lambda x: 1 if x.lower()=='м' else 0)

df2['row_number'] = df2['row_number'].fillna(df2['row_number'].max())

upper_bound = find_upper_bound(df2['liters_drunk'])
df2['liters_drunk'] = df2['liters_drunk'].map(lambda x: df2['liters_drunk'].mean() if x<0 and x<upper_bound else x)
# print(df2)

df2['drink'] = df2['drink'].str.contains('beer|пиво', case=False, na=False).astype(int)

df2['age_do_18'] = np.where((df2['age'] <= 18), 1, 0)
# df2['age_do_18'] = df2['age'].map(lambda x: 1 if x<=18 else 0)
df2['age_ot_18_do_50'] = np.where((18 < df2['age']) & (df2['age'] <= 50), 1, 0)
# df2['age_ot_18_do_50'] = df2['age'].map(lambda x: 1 if 18< x <=50 else 0)
df2['age_50+'] = np.where((df2['age'] > 50), 1, 0)
# df2['age_50+'] = df2['age'].map(lambda x: 1 if x > 50 else 0)

del df2['age']

df2 = pd.merge(df2, df, on = 'check_number')

df2['morning'] = df2['session_start'].apply(lambda x: 1 if int(x[:x.find(':')]) < 12 else 0)
df2['day'] = df2['session_start'].map(lambda x: 1 if 12 <= int(x[:x.find(':')]) < 18 else 0)
df2['evening'] = df2['session_start'].map(lambda x: 1 if 18 <= int(x[:x.find(':')]) < 24 else 0)

del df2['session_start']
print(df2)