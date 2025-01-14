# config.py
import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Base project directory
DATA_DIR = os.path.join(BASE_DIR, "data")  # Data folder
DATA_FILE_PATH = os.path.join(DATA_DIR, "Revenue totals (dummy data set).xlsx")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

TARGET_COLUMN = "Revenue"
TEST_SIZE = 0.3
RANDOM_STATE = 42

MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 0.1, 1],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}
