import typer
import pandas as pd
from typing_extensions import Annotated
from gpt.config import get_data_path, get_dataframe


def extract_abstracts(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Extracts the abstracts from the dataframe and returns a new dataframe with the abstracts.

    Args:
        df (pd.DataFrame): The input dataframe containing the abstracts.

    Returns:
        pd.DataFrame: A new dataframe containing the extracted abstracts.
    """
    # convert string "True"/"False" to actual booleans, if needed
    df[col_name] = df[col_name].astype(str).str.lower()
    # filter for True abstracts in status column
    df_abs = df[df[col_name].isin(["true", "maybe"])]
    return df_abs


def main(
    file_name: Annotated[
        str, typer.Option(help="The name of the dataset")
    ] = "dataset-updated.xlsx",
    col_name: Annotated[
        str, typer.Option(help="The name of the column containing the True abstracts")
    ] = "verdict",
    output_file_name: Annotated[
        str, typer.Option(help="The name of the output dataset with abstracts")
    ] = "abstracts.xlsx",
):
    """
    Main function to extract abstracts from the dataset.

    Args:
        file_name (str): The name of the dataset file.
    """
    # load the dataset
    df = get_dataframe(file_name=file_name)

    # extract abstracts
    df_abs = extract_abstracts(df=df, col_name=col_name)

    # save the extracted abstracts to a new file
    base_dir = get_data_path()
    output_path = f"{base_dir}/extracted_data/{output_file_name}"
    df_abs.to_excel(output_path, index=False)
    print(f"Extracted abstracts saved to {output_file_name}")
    print(f"Number of abstracts extracted: {len(df_abs)}")


# run the script:
# gpt preprocessing extract_abstracts --file-name dataset-updated.xlsx --output-file-name abstracts.xlsx
# gpt preprocessing extract_abstracts
