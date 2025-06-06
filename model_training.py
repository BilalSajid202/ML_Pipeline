import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import lightgbm as lgb
import os

class MLModelPipeline:
    """
    Class to handle training and evaluation of classification and regression models.
    Automatically saves evaluation results to outputs/evaluation_summary.txt.
    """

    def __init__(self, df: pd.DataFrame, target_column, problem_type=None, model_choice=None):
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.model_choice = model_choice
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.evaluation_path = os.path.join(self.output_dir, "evaluation_summary.txt")

        self.X, self.y = self._split_features_target()
        self._auto_detect_problem_type()
        self._set_default_model()

    def _split_features_target(self):
        if isinstance(self.target_column, (list, tuple)):
            X = self.df.drop(columns=self.target_column)
            y = self.df[self.target_column]
        else:
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
        return X, y

    def _auto_detect_problem_type(self):
        if self.problem_type is None:
            if isinstance(self.y, pd.DataFrame):
                self.problem_type = "classification"
            elif pd.api.types.is_numeric_dtype(self.y) and self.y.nunique() > 10:
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"
            print(f"ℹ️ Auto-detected problem type: {self.problem_type}")

    def _set_default_model(self):
        if self.model_choice is None:
            self.model_choice = {
                "classification": "lightgbm",
                "regression": "linear"
            }[self.problem_type]
            print(f"ℹ️ Using default model: {self.model_choice}")

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        if self.problem_type == "classification":
            return self._train_classification(X_train, X_test, y_train, y_test)
        elif self.problem_type == "regression":
            return self._train_regression(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("❌ Invalid problem type.")

    def _train_classification(self, X_train, X_test, y_train, y_test):
        models = {
            "logistic": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1),
            "svm": SVC(),
            "lightgbm": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42)
        }

        if self.model_choice not in models:
            raise ValueError(f"❌ Invalid classification model choice: {self.model_choice}")

        model = models[self.model_choice]
        model.fit(X_train, y_train)
        self._evaluate_classification(model, X_test, y_test)
        return model

    def _train_regression(self, X_train, X_test, y_train, y_test):
        models = {
            "linear": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(),
            "random_forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
            "svm": SVR()
        }

        if self.model_choice not in models:
            raise ValueError(f"❌ Invalid regression model choice: {self.model_choice}")

        model = models[self.model_choice]
        model.fit(X_train, y_train)
        self._evaluate_regression(model, X_test, y_test)
        return model

    def _evaluate_classification(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        print("\n✅ Classification Report")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        self._save_metrics_to_file(metrics, "Classification")

    def _evaluate_regression(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2 Score": r2_score(y_test, y_pred)
        }

        print("\n✅ Regression Report")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        self._save_metrics_to_file(metrics, "Regression")

    def _save_metrics_to_file(self, metrics: dict, title: str):
        """Write evaluation metrics to text file."""
        with open(self.evaluation_path, "w") as f:
            f.write(f"{title} Evaluation Report\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                if isinstance(value, list):
                    f.write(f"{key}:\n{value}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
        print(f"\n📁 Evaluation summary saved to {self.evaluation_path}")
