import os
import pandas as pd
from pathlib import Path


def get_data_path():
    home = Path(os.path.expanduser("~"))
    return f"{home}/github/gpt-sandbox/data"


def get_dataframe(file_name: str = "dataset-updated.xlsx"):
    data_path = f"{get_data_path()}/{file_name}"
    df = pd.read_excel(data_path)
    return df
