import pandas as pd
from pathlib import Path


def get_data_path():
    """Get the base project directory."""
    return Path(__file__).resolve().parent.parent


def get_dataframe(file_name: str = "dataset-updated.xlsx"):
    data_path = f"{get_data_path()}/{file_name}"
    df = pd.read_excel(data_path)
    return df
