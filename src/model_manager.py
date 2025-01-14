import optuna
from src.utils import export_and_archive
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

class ModelManager:
    def __init__(self, model_name):
        """
        Initialize the model manager with a specific model.
        """
        self.model_name = model_name
        self.model = None
        self.study = None

    def _objective(self, trial, X_train, y_train, X_test, y_test):
        """
        Objective function for Optuna optimization.
        """
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        # Initialize and train the model
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        # Predict and calculate MSE
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    def optimize_model(self, X_train, y_train, X_test, y_test, n_trials=50):
        """
        Optimize model hyperparameters using Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

        print(f"Best hyperparameters: {study.best_params}")
        self.model = XGBRegressor(**study.best_params, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test, export_dir="exports"):
        """
        Evaluate the model and return metrics.
        """
        y_pred = self.model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        export_and_archive(results, "predictions", export_dir=export_dir, file_type="csv", archive=True)
        return rmse, y_pred
