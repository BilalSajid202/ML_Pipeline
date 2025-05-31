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


class MLModelPipeline:
    """
    Class to handle training and evaluation of classification and regression models.
    Supports several sklearn models and LightGBM for classification.
    """

    def __init__(self, df: pd.DataFrame, target_column, problem_type=None, model_choice=None):
        """
        Initialize pipeline with dataset, target column, problem type, and model choice.
        
        :param df: Input DataFrame containing features and target.
        :param target_column: Name or list of names of the target variable(s).
        :param problem_type: "classification" or "regression". Auto-detected if None.
        :param model_choice: Model to use. Defaults to LightGBM for classification,
                             Linear Regression for regression if None.
        """
        self.df = df
        self.target_column = target_column
        self.problem_type = problem_type
        self.model_choice = model_choice

        self.X, self.y = self._split_features_target()
        self._auto_detect_problem_type()
        self._set_default_model()

    def _split_features_target(self):
        """Split dataframe into features (X) and target (y)."""
        if isinstance(self.target_column, (list, tuple)):
            X = self.df.drop(columns=self.target_column)
            y = self.df[self.target_column]
        else:
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
        return X, y

    def _auto_detect_problem_type(self):
        """Automatically determine if problem is classification or regression."""
        if self.problem_type is None:
            if isinstance(self.y, pd.DataFrame):
                self.problem_type = "classification"
            elif pd.api.types.is_numeric_dtype(self.y) and self.y.nunique() > 10:
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"
            print(f"ℹ️ Auto-detected problem type: {self.problem_type}")

    def _set_default_model(self):
        """Set default model choice if not specified."""
        if self.model_choice is None:
            self.model_choice = {
                "classification": "lightgbm",
                "regression": "linear"
            }[self.problem_type]
            print(f"ℹ️ Using default model: {self.model_choice}")

    def run(self):
        """Run the entire pipeline: train, evaluate and return trained model."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        if self.problem_type == "classification":
            return self._train_classification(X_train, X_test, y_train, y_test)
        elif self.problem_type == "regression":
            return self._train_regression(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("❌ Invalid problem type.")

    # --- Training methods for classification ---
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

    # --- Training methods for regression ---
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

    # --- Evaluation methods ---
    def _evaluate_classification(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("\n✅ Classification Report")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
        print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    def _evaluate_regression(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("\n✅ Regression Report")
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))
        print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
        print("R2 Score:", r2_score(y_test, y_pred))
