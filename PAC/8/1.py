import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_columns', None)

def parse_date(col, df_to, df_fr):
    date = pd.to_datetime(df_fr[col])
    df_to['day_'+col] = date.dt.day
    df_to['month_'+col] = date.dt.month
    df_to['year_'+col] = date.dt.year

df = pd.read_csv('wells_info_with_prod.csv')

new_df = pd.DataFrame()
new_df['Prod1Year'] = df['Prod1Year']
parse_date('PermitDate', new_df, df)
parse_date('SpudDate', new_df, df)
parse_date('CompletionDate', new_df, df)
new_df['days_spad_to_compl'] = (new_df['year_CompletionDate'] - new_df['year_SpudDate']) * 365 + (new_df['month_CompletionDate'] - new_df['month_SpudDate']) * 30 + (new_df['day_CompletionDate'] - new_df['day_SpudDate'])

new_df['LATERAL_LENGTH_BLEND'] = df['LATERAL_LENGTH_BLEND']
new_df['PROP_PER_FOOT'] = df['PROP_PER_FOOT']
new_df['WATER_PER_FOOT'] = df['WATER_PER_FOOT']
new_df['LatWGS84'] = df['LatWGS84']
new_df['LonWGS84'] = df['LonWGS84']
# print(df['formation'].unique())
uniq = df['formation'].unique()
new_df['formation'] = df['formation'].map(lambda x: np.where(uniq == x)[0][0])
print(new_df.head())

X = new_df.drop('Prod1Year', axis=1)
y = new_df['Prod1Year']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))[:,0]
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))[:,0]
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))[:,0]

