import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier

def fix_nan(df):
    if df.size*0.01 < df.isnull().sum().sum():
        df.fillna(df.mean(), inplace=True)
    else:
        df.dropna(inplace=True)

def new_columns(df):
    df["annual_income2"] = df["annual_income"]**2
    df["credit_score2"] = df["credit_score"]**2


def select_features(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    })

    # print(feature_importance_df)

    important_features = feature_importance_df[
        feature_importance_df['importance'] > 0.001
        ]['feature'].tolist()

    return important_features

def metrics(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
    }

    return metrics


df = pd.read_csv(r".\train.csv")

fix_nan(df)


y = df["loan_paid_back"]
df.drop("loan_paid_back", axis=1, inplace=True)

print(df.size,df.isnull().sum().sum())

dummy = pd.DataFrame()

for i in df.columns:
    if df[i].dtype == "object":
        dummy = pd.concat([dummy,pd.get_dummies(df[i], drop_first=True,prefix=i, dtype="int")], axis=1)
        df = df.drop(i,axis=1)

df = pd.concat([df,dummy], axis=1)
# # print(df)
#
new_columns(df)
#
# corr_matrix = df.corr()
# Визуализация
# plt.figure(figsize=(13, 12))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', xticklabels=False, yticklabels=False)
# plt.title('Матрица корреляции')
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train = pd.DataFrame(X_train, columns=df.columns)

scaler = StandardScaler()
selected = select_features(X_train, y_train)

X_train_selected = X_train[selected]
X_val_selected = X_val[selected]
X_test_selected = X_test[selected]

X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_val_selected_scaled = scaler.transform(X_val_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

models = [RandomForestClassifier(),
          KNeighborsClassifier(),
          DecisionTreeClassifier(), XGBClassifier(),
          GradientBoostingClassifier(), LogisticRegression(max_iter=1000), GaussianNB()]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metric = metrics(y_test, y_pred)
    print(metric)
#
params = {
    'n_estimators': [100, 200],           # 2 значения
    'max_depth': [3, 5],                  # 2 значения
    'learning_rate': [0.05, 0.1],         # 2 значения
    'subsample': [0.8],                   # 1 значение
    'colsample_bytree': [0.8],            # 1 значение
    'reg_alpha': [0, 0.1],                # 2 значения
    'reg_lambda': [1],                    # 1 значение
    'gamma': [0],                         # 1 значение
}
# #
metr = {
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall'
}
#
cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#
model = XGBClassifier()

grid = GridSearchCV(estimator=model,param_grid=params, scoring=metr, cv=cv_kfold ,n_jobs=-1, verbose=1, refit = 'roc_auc')

grid.fit(X_train_selected_scaled, y_train)

y_pred = grid.predict(X_test_selected_scaled)
metric = metrics(y_test, y_pred)
print(metric)
# # print(grid.)


