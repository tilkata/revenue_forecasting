from src.utils import export_and_archive

def create_features(data,export_dir="exports"):
    """
    Create additional features for revenue prediction.
    """
    # Lagging features
    data["Lag_1"] = data["Revenue"].shift(1)
    data["Lag_6"] = data["Revenue"].shift(6)

    # Rolling averages
    data["MA_6"] = data["Revenue"].rolling(window=6).mean()

    # Drop rows with NaN values caused by lagging/rolling features
    data = data.dropna().reset_index(drop=True)

    export_and_archive(data, "processed_features", export_dir=export_dir, file_type="csv", archive=True)

    return data
