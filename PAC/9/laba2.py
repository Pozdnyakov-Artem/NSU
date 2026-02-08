from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class MyRandomForest:
    def __init__(self, number_trees=200):
        self.number_trees = number_trees
        self.trees = []
        self.selected_features = []

    def fit(self, x, y):
        for i in range(self.number_trees):
            len_x = len(x)

            n_features = x.shape[1]
            n_selected_features = int(np.sqrt(n_features))

            ind = np.random.choice(len_x, size = len_x, replace = True)
            x_bootstrap = x.iloc[ind] if hasattr(x, 'iloc') else x[ind]
            y_bootstrap = y.iloc[ind] if hasattr(y, 'iloc') else y[ind]

            feature_indices = np.random.choice(
                n_features,
                size=n_selected_features,
                replace=False
            )

            self.selected_features.append(feature_indices)

            x_selected = x_bootstrap.iloc[:, feature_indices] if hasattr(x_bootstrap, 'iloc') else x_bootstrap[:,
                                                                                                   feature_indices]

            self.trees.append(DecisionTreeClassifier(max_depth = 10))
            self.trees[-1].fit(x_bootstrap, y_bootstrap)
    def predict(self, x):
        predictions = []
        ans = []
        for i in range(self.number_trees):
            selected_features = x.iloc[:, self.selected_features[i]] if hasattr(x, 'iloc') else x[:, self.selected_features[i]]
            predictions.append(self.trees[i].predict(x))

        for i in range(len(predictions[0])):
            var = [predictions[j][i] for j in range(len(predictions))]
            most_common = Counter(var).most_common(1)[0][0]
            ans.append(most_common)
        return ans

df = pd.read_csv("titanic_prepared.csv")
X = df.drop(["label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test =  train_test_split(X,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)

model = MyRandomForest()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(f"Точность моего леса: {accuracy_score(y_test, y_pred)}")

model = DecisionTreeClassifier(max_depth=10)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(f"Точность одного дерева: {accuracy_score(y_test, y_pred)}")