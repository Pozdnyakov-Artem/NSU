import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def cool_cola(model, X_train_scaled, X_test_scaled, y_train, y_test, top_features):

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print(f"Точность на всех: {accuracy_score(y_test, y_pred)}")

    X_train_scaled_selected = X_train_scaled[top_features]
    X_test_scaled_selected = X_test_scaled[top_features]

    model.fit(X_train_scaled_selected, y_train)

    y_pred = model.predict(X_test_scaled_selected)

    print(f"Точность на 2: {accuracy_score(y_test, y_pred)}")

df = pd.read_csv("titanic_prepared.csv")

X = df.drop("label", axis = 1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
top_features = feature_importance_df['feature'].values[:2]

print("DecisionTreeClassifier")
cool_cola(DecisionTreeClassifier(max_depth=3, criterion='entropy'), X_train_scaled, X_test_scaled, y_train, y_test, top_features)

print("LogisticRegression")
cool_cola(LogisticRegression(random_state=42), X_train_scaled, X_test_scaled, y_train, y_test, top_features)

print("XGBClassifier")
cool_cola(XGBClassifier(random_state=42), X_train_scaled, X_test_scaled, y_train, y_test, top_features)