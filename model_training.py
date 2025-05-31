from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import pandas as pd
import lightgbm as lgb

# CLASSIFICATION MODELS
def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return evaluate_classification(model, X_test, y_test)

def train_decision_tree_classifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return evaluate_classification(model, X_test, y_test)

def train_random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    return evaluate_classification(model, X_test, y_test)

def train_svm_classifier(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    return evaluate_classification(model, X_test, y_test)

def train_lightgbm_classifier(X_train, X_test, y_train, y_test):
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return evaluate_classification(model, X_test, y_test)

# REGRESSION MODELS
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return evaluate_regression(model, X_test, y_test)

def train_decision_tree_regressor(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return evaluate_regression(model, X_test, y_test)

def train_random_forest_regressor(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    return evaluate_regression(model, X_test, y_test)

def train_svm_regressor(X_train, X_test, y_train, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    return evaluate_regression(model, X_test, y_test)

# EVALUATION FUNCTIONS
def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n✅ Classification Report")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return model

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n✅ Regression Report")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 Score:", r2_score(y_test, y_pred))
    return model

# MAIN PIPELINE
def run_model_pipeline(df, target_column, problem_type=None, model_choice=None):
    if isinstance(target_column, (list, tuple)):
        X = df.drop(columns=target_column)
        y = df[target_column]
    else:
        X = df.drop(columns=[target_column])
        y = df[target_column]

    if problem_type is None:
        if isinstance(y, pd.DataFrame):
            problem_type = "classification"
        elif pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            problem_type = "regression"
        else:
            problem_type = "classification"
        print(f"ℹ️ Auto-detected problem type: {problem_type}")

    if model_choice is None:
        model_choice = {
            "classification": "lightgbm",
            "regression": "linear"
        }[problem_type]
        print(f"ℹ️ Using default model: {model_choice}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem_type == "classification":
        if model_choice == "logistic":
            return train_logistic_regression(X_train, X_test, y_train, y_test)
        elif model_choice == "decision_tree":
            return train_decision_tree_classifier(X_train, X_test, y_train, y_test)
        elif model_choice == "random_forest":
            return train_random_forest_classifier(X_train, X_test, y_train, y_test)
        elif model_choice == "svm":
            return train_svm_classifier(X_train, X_test, y_train, y_test)
        elif model_choice == "lightgbm":
            return train_lightgbm_classifier(X_train, X_test, y_train, y_test)

    elif problem_type == "regression":
        if model_choice == "linear":
            return train_linear_regression(X_train, X_test, y_train, y_test)
        elif model_choice == "decision_tree":
            return train_decision_tree_regressor(X_train, X_test, y_train, y_test)
        elif model_choice == "random_forest":
            return train_random_forest_regressor(X_train, X_test, y_train, y_test)
        elif model_choice == "svm":
            return train_svm_regressor(X_train, X_test, y_train, y_test)

    raise ValueError("❌ Invalid problem type or model choice.")
