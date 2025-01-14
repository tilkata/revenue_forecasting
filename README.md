# Revenue Forecasting

This repository contains a machine learning pipeline designed to forecast subscription-based revenue using the XGBoost model and Random Decision Forest. The project aims to provide accurate revenue predictions by leveraging advanced machine learning techniques and feature engineering.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data Files](#data-files)
- [Contributing](#contributing)
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

2. Feature engineering:
   - The `create_features` function in `src/feature_engineering.py` creates additional features for revenue prediction.

3. Model management:
   - The `ModelManager` class in `src/model_manager.py` handles model optimization and evaluation using XGBoost and Optuna.

4. Visualization:
   - The `VisualizationManager` class in `src/visualization_manager.py` provides functions to plot predictions and feature importance.

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
