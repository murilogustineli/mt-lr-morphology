import os
import re
import typer
import pandas as pd

from openai import OpenAI
from typing_extensions import Annotated
from dotenv import load_dotenv

from gpt.config import get_data_path


def prompt_gpt(
    client,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 200,
):
    # query ChatGPT
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to carefully analyze academic abstracts based on specific inclusion and exclusion criteria.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content


def import_data(file_name: str):
    """
    Imports data from a CSV file.
    """
    data_path = get_data_path()
    input_path = f"{data_path}/generated_answers/{file_name}"
    df = pd.read_csv(input_path)
    return df


def get_rq_patterns():
    """
    Returns the regular expressions for extracting answers.
    """
    rq_patterns = {
        "rq1": r"1\. Challenges of morphological complexity\s+Answer: (.*?)\s+Evidence:",
        "rq2": r"2\. Proposed techniques\s+Answer: (.*?)\s+Evidence:",
        "rq3": r"3\. Morphology-aware techniques\s+Answer: (.*?)\s+Evidence:",
        "rq4": r"4\. Specific findings per morphological typology.*?Answer: (.*?)\s+Evidence:",
    }
    return rq_patterns


# function to extract answers based on patterns
def extract_answers(row, patterns):
    answers = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, row)
        answers[key] = match.group(1) if match else None
    return answers


def combine_answers(answers: list) -> str:
    combined = ", ".join(answers)
    return combined


def get_prompt(answer: str):
    prompt = f"""
    Based on the following analysis, identify five **thematic categories** that summarize the key concepts, methods, or contributions mentioned.

    --- BEGIN ANALYSIS ---
    {answer}
    --- END ANALYSIS ---

    The categories should reflect actual *topics or themes* found in the answers, such as techniques, challenges, or typologies discussed (e.g., "Cross-lingual Transfer Learning", "Morphological Generation", "Polysynthetic Language Challenges").

    Only return the categories, comma-separated, in the following format:
    OUTPUT: Category 1, Category 2, Category 3, Category 4, Category 5
    """
    return prompt


def extract_output(result: str) -> tuple:
    # Use re.DOTALL to make '.' match newline characters
    match = re.search(r"OUTPUT:\s*(.*)", result, re.DOTALL)
    if match:
        output_text = match.group(1).replace("\n", " ").strip().strip("*")
        return output_text
    else:
        return result


def workflow(
    model: str = "gpt-4o-mini",
    dataset_name: str = "limit_n=10_text_answers_gpt-4o-mini.csv",
    temperature: float = 0.0,
    max_tokens: int = 500,
):
    """
    Workflow function to generate answers for full-text or abstract papers.
    """

    # load environment variables
    load_dotenv()

    # get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    # init OpenAI client
    client = OpenAI(api_key=api_key)

    # import dataframe
    df = import_data(dataset_name)

    # remove any rows with NaN values in the 'generated_answers' column
    df = df.dropna(subset=["generated_answers"])

    # apply the extraction function to each row
    answers_df = df["generated_answers"].apply(
        lambda x: extract_answers(x, get_rq_patterns())
    )
    # convert the list of answers into separate columns
    answers_df = pd.DataFrame(answers_df.tolist(), index=df.index)
    # merge the extracted answers with the original dataframe
    df = df.drop(columns=[col for col in df.columns if col.startswith("rq")])
    df = pd.concat([df, answers_df], axis=1)
    # print the columns names
    print(f"Columns in the dataframe: {df.columns}")

    # extract the answers into separate lists
    rq1_answers = df["rq1"].tolist()
    rq2_answers = df["rq2"].tolist()
    rq3_answers = df["rq3"].tolist()
    rq4_answers = df["rq4"].tolist()

    # combine all answers into a single string
    rq1_answers_combined = combine_answers(rq1_answers)
    rq2_answers_combined = combine_answers(rq2_answers)
    rq3_answers_combined = combine_answers(rq3_answers)
    rq4_answers_combined = combine_answers(rq4_answers)

    responses = {}
    combined_answers = [
        rq1_answers_combined,
        rq2_answers_combined,
        rq3_answers_combined,
        rq4_answers_combined,
    ]
    # loop through each answer and query ChatGPT
    for i, answer in enumerate(combined_answers):
        prompt = get_prompt(answer)
        # query ChatGPT
        response = prompt_gpt(
            client, prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
        response = extract_output(response)
        responses[f"rq{i+1}_categories"] = response
        print(f"Response {i+1}/{len(combined_answers)}...")

    # create a DataFrame to store the responses
    responses_df = pd.DataFrame([responses])
    # save the results to a CSV file
    dataset_name = dataset_name.split("_gpt")[0]
    dataset_name = f"{dataset_name}_{model}.csv"
    data_path = f"{get_data_path()}/categories/{dataset_name}"
    responses_df.to_csv(data_path, index=False)
    print(f"Results saved to {data_path}")


def main(
    model: Annotated[
        str, typer.Option(help="The model to use for the analysis")
    ] = "gpt-4o-mini",
    dataset_name: Annotated[
        str, typer.Option(help="The name of the dataset")
    ] = "limit_n=30_text_answers_gpt-4o.csv",
    temperature: Annotated[
        float, typer.Option(help="The temperature for sampling")
    ] = 0.0,
    max_tokens: Annotated[
        int, typer.Option(help="The maximum number of tokens to generate")
    ] = 500,
):
    """
    Main function to run the workflow.
    """
    workflow(
        model=model,
        dataset_name=dataset_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


"""
Before running the script, make sure to run the generate_answers.py script
to generate the answers to the questions, and use the same generated dataset as
the dataset_name input for this script.
"""

# models: gpt-4o-mini, gpt-4o
# run the script in the terminal:
# gpt categorization categorize --model gpt-4o-mini --dataset-name limit_n=30_text_answers_gpt-4o.csv
