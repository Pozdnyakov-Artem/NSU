import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def prep(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df_cl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    return pd.concat((df_num, df_sex, df_emb, df_cl), axis = 1)

def fit_model(model,param_grid, X_train_scaled, X_test_scaled,
                         y_train, y_test, top_features, model_name):

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.score(X_test_scaled, y_test)

    accuracy = []
    accuracy.append(y_pred)

    for i in [2,4,8]:
        X_train_selected = X_train_scaled[top_features[:i]]
        X_test_selected = X_test_scaled[top_features[:i]]

        grid_search_selected = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_selected.fit(X_train_selected, y_train)

        best_model_selected = grid_search_selected.best_estimator_
        y_pred = best_model_selected.score(X_test_selected, y_test)
        accuracy.append(y_pred)

    print(f"Точность на всех данных: {accuracy[0]}, на 2: {accuracy[1]}, на 4: {accuracy[2]}, на 8: {accuracy[3]}")

df = pd.read_csv(r'train.csv')

df_val = df["Survived"]
df_train = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)

df_train = prep(df_train)

df_train = df_train.fillna(df_train.median())

X_train, X_test, y_train, y_test = train_test_split(df_train, df_val, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train_scaled, y_train)
accuracy = []
y_pred = model.predict(X_test_scaled)
accuracy.append(accuracy_score(y_test, y_pred))

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
top_features = feature_importance_df['feature'].values

param_grids = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

print("RandomForestClassifier")
fit_model(
    RandomForestClassifier(random_state=42),
    param_grids['RandomForest'],
    X_train_scaled, X_test_scaled,
    y_train, y_test, top_features,
    "RandomForest"
)

print("LogisticRegression")
fit_model(
    LogisticRegression(max_iter=1000),
    param_grids['LogisticRegression'],
    X_train_scaled, X_test_scaled,
    y_train, y_test, top_features,
    "LogisticRegression"
)
#
print("KNeighborsClassifier")
fit_model(
    KNeighborsClassifier(),
    param_grids['KNeighbors'],
    X_train_scaled, X_test_scaled,
    y_train, y_test, top_features,
    "KNeighbors"
)

#
print("XGBClassifier")
fit_model(
    XGBClassifier(),
    param_grids['XGBoost'],
    X_train_scaled, X_test_scaled,
    y_train, y_test, top_features,
    "XGBoost"
)