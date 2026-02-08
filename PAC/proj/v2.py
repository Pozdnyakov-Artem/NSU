import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, \
    StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def load_and_preprocess():
    df = pd.read_csv(r".\train.csv")
    df.drop(["id"], axis=1, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def dummies(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df

def cor(X):
    plt.figure(figsize=(10, 8))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=False, yticklabels=False)
    plt.title('Матрица корреляции признаков')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

    print("\n=== Топ 10 коррелирующих пар признаков ===")
    corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
    top_pairs = corr_pairs[corr_pairs < 1].head(10)
    print(top_pairs)

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_tri.columns
               if any(abs(upper_tri[column]) > 0.9)]
    print("Удалённые после корреляции:",to_drop)
    return X.drop(columns=to_drop)


def create_polynomial_features(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(X[numeric_cols])

    poly_feature_names = poly.get_feature_names_out(X[numeric_cols].columns)
    X_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)

    X_combined = pd.concat([X.drop(columns=numeric_cols), X_poly], axis=1)

    print(f"\n=== Добавление полиномиальных признаков ===")
    print(f"Исходное количество признаков: {X.shape[1]}")
    print(f"После добавления полиномиальных: {X_combined.shape[1]}")
    print(f"Добавлено {X_combined.shape[1] - X.shape[1]} новых признаков")

    print("\nПримеры новых признаков:")
    new_features = [col for col in X_combined.columns if '^2' in col or ' ' in col]
    print(f"Квадратные признаки: {[col for col in new_features if '^2' in col][:5]}")
    print(f"Признаки взаимодействия: {[col for col in new_features if ' ' in col][:5]}")

    return X_combined

def plot_feature_importance(X_train, y_train, model_name="Random Forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_

    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    })

    # print(feature_importance_df)

    important_features = feature_importance_df[
        feature_importance_df['importance'] > 0.001
        ]['feature'].tolist()

    print("import", important_features)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(f'Важность признаков ({model_name})')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]])
    plt.xticks(range(min(20, len(importances))),
               X_train.columns[indices[:20]], rotation=90)
    plt.ylabel('Важность')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

    return important_features


def train_base_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)

    print("\n=== Сравнение моделей (параметры по умолчанию) ===")
    print(results_df.to_string(index=False))

    return results_df

def hyperparameter_tuning(X_train, y_train):
    params = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9], # количество случайных строк
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9], # количество признаков
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 1, 5, 10],
        'gamma': [0, 0.1, 0.5, 1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        estimator=XGBClassifier(),
        param_distributions=params,
        n_iter=64,
        scoring={
            'accuracy': 'accuracy',
            'f1': 'f1',
            'recall': 'recall',
            'roc_auc': 'roc_auc'
        },
        refit='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Результаты GridSearchCV ===")
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая ROC-AUC score: {grid_search.best_score_:.4f}")

    return grid_search


def plot_model_results(y_test, y_pred, y_pred_proba=None, model_name=""):

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'Матрица ошибок - {model_name}')
    axes[0, 0].set_xlabel('Предсказанный класс')
    axes[0, 0].set_ylabel('Истинный класс')

    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        axes[0, 1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve - {model_name}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision, marker='.')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title(f'Precision-Recall Curve - {model_name}')
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(0.1, 0.5, f'Метрики модели {model_name}:\n\n'
                              f'Accuracy: {accuracy_score(y_test, y_pred):.3f}\n'
                              f'Precision: {precision_score(y_test, y_pred):.3f}\n'
                              f'Recall: {recall_score(y_test, y_pred):.3f}\n'
                              f'F1-Score: {f1_score(y_test, y_pred):.3f}\n'
                              f'ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}',
                    fontsize=12, bbox=dict(boxstyle="round", alpha=0.1))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'model_results_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.show()

    print(f"\n=== Classification Report - {model_name} ===")
    print(classification_report(y_test, y_pred))


def main():
    print("Шаг 1: Загрузка и предобработка данных")
    df = load_and_preprocess()

    print("\nШаг 2: Работа с признаками")
    df = dummies(df)
    y = df["loan_paid_back"]
    df = df.drop("loan_paid_back", axis=1)
    X_poly = create_polynomial_features(df)
    X_poly = cor(X_poly)

    print(X_poly.columns)

    print("\nШаг 3: Дробление")

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nШаг 4: Анализ важности признаков")
    important_features = plot_feature_importance(pd.DataFrame(X_train_scaled, columns=X_poly.columns), y_train)

    X_train_important = pd.DataFrame(X_train_scaled, columns=X_poly.columns)[important_features]
    X_test_important = pd.DataFrame(X_test_scaled, columns=X_poly.columns)[important_features]

    print("\nШаг 5: Обучение моделей с параметрами по умолчанию")
    results_df = train_base_models(X_train_important, X_test_important, y_train, y_test)

    print("\nШаг 6: Поиск оптимальных гиперпараметров")
    best_model_name = results_df.iloc[0]['Model']
    print(f"Лучшая модель: {best_model_name}")

    grid_search = hyperparameter_tuning(X_train_important, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_important)
    y_pred_proba = best_model.predict_proba(X_test_important)[:, 1]

    print("\nШаг 7: Визуализация результатов")
    plot_model_results(y_test, y_pred, y_pred_proba, best_model_name)

    print("\n=== Результаты кросс-валидации ===")
    cv_scores = cross_val_score(best_model, X_train_important, y_train,
                                cv=5, scoring='roc_auc')
    print(f"ROC-AUC кросс-валидация: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Model'], results_df['ROC-AUC'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ROC-AUC Score')
    plt.title('Сравнение моделей по ROC-AUC')
    plt.tight_layout()
    plt.savefig('models_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()