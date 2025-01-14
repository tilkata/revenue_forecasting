import os
import pandas as pd
from datetime import datetime
import json

def export_and_archive(data, file_name, export_dir="exports", file_type="csv", archive=True):
    """
    Export and archive data to the specified directory and file format.
    """
    os.makedirs(export_dir, exist_ok=True)

    # Define file paths
    static_file_path = os.path.join(export_dir, f"{file_name}.{file_type}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_file_path = os.path.join(export_dir, f"{file_name}_{timestamp}.{file_type}")

    try:
        if isinstance(data, pd.DataFrame):
            if file_type == "csv":
                data.to_csv(static_file_path, index=False)
                if archive:
                    data.to_csv(archive_file_path, index=False)
            elif file_type == "json":
                data.to_json(static_file_path, orient="records", lines=True)
                if archive:
                    data.to_json(archive_file_path, orient="records", lines=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        elif isinstance(data, dict):
            with open(static_file_path, "w") as f:
                json.dump(data, f, indent=4)
            if archive:
                with open(archive_file_path, "w") as f:
                    json.dump(data, f, indent=4)
        else:
            raise ValueError("Unsupported data type. Must be DataFrame or dict.")

        print(f"Exported to {static_file_path}")
        if archive:
            print(f"Archived to {archive_file_path}")

    except Exception as e:
        print(f"Error during export: {e}")
