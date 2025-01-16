# Revenue Forecasting

This repository contains a machine learning pipeline designed to forecast subscription-based revenue using the XGBoost model and Random Decision Forest. The project aims to provide accurate revenue predictions by using advanced machine learning techniques and feature engineering.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data Files](#data-files)
- [Additional Resources](#additional-resources)

## Project Overview

The goal of this project is to forecast subscription-based revenue using machine learning models. The pipeline includes data processing, feature engineering, model optimization, and visualization. The primary models used in this project are XGBoost and Random Decision Forest, with Optuna for hyperparameter optimization.

## Setup and Installation

To set up the environment and install dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tilkata/revenue_forecasting.git
   cd revenue_forecasting
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

To use the provided scripts for data processing, feature engineering, model management, and visualization, follow these steps:

1. Load and preprocess data:
   - The `load_and_clean_data` function in `src/data_processing.py` loads and cleans the data from the specified file path.
   - **Example**:
     ```python
     from src.data_processing import load_and_clean_data
     data = load_and_clean_data('path/to/data.csv')
     ```

2. Feature engineering:
   - The `create_features` function in `src/feature_engineering.py` creates additional features for revenue prediction.
   - **Example**:
     ```python
     from src.feature_engineering import create_features
     features = create_features(data)
     ```

3. Model management:
   - The `ModelManager` class in `src/model_manager.py` handles model optimization and evaluation using XGBoost and Optuna.
   - **Example**:
     ```python
     from src.model_manager import ModelManager
     model_manager = ModelManager(model_name="xgboost")
     model_manager.optimize_model(X_train, y_train, X_test, y_test)
     predictions = model_manager.evaluate_model(X_test, y_test)
     ```

4. Visualization:
   - The `VisualizationManager` class in `src/visualization_manager.py` provides functions to plot predictions and feature importance.
   - **Example**:
     ```python
     from src.visualization_manager import VisualizationManager
     viz_manager = VisualizationManager()
     viz_manager.plot_predictions(y_test, predictions)
     viz_manager.plot_feature_importance(model_manager.model, feature_names)
     ```

### Changing the Model

To change the model used for revenue prediction, you can update the `model_name` parameter when initializing the `ModelManager` class. For example, to use a Random Forest model:

```python
from src.model_manager import ModelManager

# Initialize ModelManager with Random Forest
model_manager = ModelManager(model_name="random_forest")
model_manager.optimize_model(X_train, y_train, X_test, y_test)
predictions = model_manager.evaluate_model(X_test, y_test)
```


### Adding Another Feature

To add another feature for revenue prediction, you can update the `create_features` function in `src/feature_engineering.py` to include the new feature. For example, to add a feature based on the day of the week:

```python
from src.feature_engineering import create_features

def create_features(data):
    # Existing feature engineering steps
    data['DayOfWeek'] = data['Bookingsdata'].dt.dayofweek
    return data
```

## Data Files

The `exports` directory contains data files generated during the pipeline execution. These files include:

- `exports/predictions.csv`: Contains the actual and predicted revenue values.
- `exports/predictions_20250114_144335.csv`: Contains the actual and predicted revenue values with a timestamp.
- `exports/processed_features.csv`: Contains the processed features used for modeling.
- `exports/processed_features_20250114_144317.csv`: Contains the processed features with a timestamp.

## Additional Resources

For more information and additional resources, refer to the following links:

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
