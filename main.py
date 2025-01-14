from src.config import DATA_FILE_PATH
from src.data_processing import load_and_clean_data, scale_features
from src.feature_engineering import create_features
from src.model_manager import ModelManager
from src.visualization_manager import VisualizationManager
from sklearn.model_selection import train_test_split
import optuna.visualization as viz

def main():
    # Step 1: Load and preprocess data
    data, _ = load_and_clean_data(DATA_FILE_PATH)

    # Step 2: Feature engineering
    data = create_features(data)

    # Step 3: Prepare data for modeling
    features = data.drop(columns=["Revenue", "Bookingsdata"])
    target = data["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Step 4: Optimize the model
    model_name = "xgboost"  # Define model name
    model_manager = ModelManager(model_name=model_name)
    print("Optimizing model with Optuna...")
    model_manager.optimize_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # Step 5: Evaluate the model
    print("Evaluating model...")
    rmse, y_pred = model_manager.evaluate_model(X_test_scaled, y_test)
    print(f"RMSE: {rmse}")

    # Step 7: Visualize results
    viz_manager = VisualizationManager()
    viz_manager.plot_predictions(y_test, y_pred)

    # Visualize optimization results if study is available
    if model_manager.study is not None:
        print("Visualizing optimization results...")
        optimization_history_fig = viz.plot_optimization_history(model_manager.study)
        optimization_history_fig.show()

        param_importances_fig = viz.plot_param_importances(model_manager.study)
        param_importances_fig.show()
    else:
        print("No optimization study available for visualization.")

if __name__ == "__main__":
    main()
