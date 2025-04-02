import pandas as pd
from pathlib import Path


def get_data_path():
    """Get the base project directory."""
    return Path(__file__).resolve().parent.parent / "data"


def get_dataframe(
    data_path: str = "dataset/dataset-updated.xlsx", read_csv: bool = False
) -> pd.DataFrame:
    input_path = f"{get_data_path()}/{data_path}"
    if read_csv:
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
    return df
