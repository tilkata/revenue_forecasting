import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationManager:
    def plot_predictions(self, y_test, y_pred):
        """
        Plot actual vs predicted values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Actual Values")
        plt.ylabel("Predictions")
        plt.title("Predictions vs Actual Values")
        plt.grid()
        plt.show()

    def plot_feature_importance(self, model, feature_names):
        """
        Plot feature importance for a model.
        """
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.grid()
        plt.show()
