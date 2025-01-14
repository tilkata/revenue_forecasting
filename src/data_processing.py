import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(file_path):
    """
    Load and clean data from the specified file path.
    """
    data = pd.ExcelFile(file_path).parse("Dummy data")

    # Drop irrelevant or redundant columns
    data = data.drop(columns=[col for col in ["Unnamed: 8"] if col in data.columns], errors="ignore")

    # Drop rows with missing values
    data = data.dropna()

    # Encode categorical columns
    categorical_columns = ["Customer", "Subscription type", "Revenuetype", "Description"]
    label_encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    # Convert dates and extract temporal features
    data["Bookingsdata"] = pd.to_datetime(data["Bookingsdata"], errors="coerce")
    data["Year"] = data["Bookingsdata"].dt.year
    data["Month"] = data["Bookingsdata"].dt.month

    return data, label_encoders


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
