import os
import re
import typer
import pandas as pd

from typing_extensions import Annotated
from dotenv import load_dotenv
from openai import OpenAI

from gpt.config import get_data_path, get_dataframe
from gpt.categorization.prompt import get_research_questions, chain_of_thought_prompt


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


# query ChatGPT
def query_gpt(
    client,
    model,
    abstract: str,
    temperature: float = 0.0,
    max_tokens: int = 250,
) -> pd.DataFrame:
    prompt = chain_of_thought_prompt(abstract, get_research_questions())
    response = prompt_gpt(client, prompt, model, temperature, max_tokens)
    return response


def extract_output_and_reasoning(result: str) -> tuple:
    # Use re.DOTALL to make '.' match newline characters
    match = re.search(r"OUTPUT:\s*\**(.*?)\**\s*REASONING:\s*(.*)", result, re.DOTALL)
    if match:
        output_text = match.group(1).replace("\n", " ").strip().strip("*")
        reasoning_text = match.group(2).replace("\n", " ").strip()
        return output_text, reasoning_text
    else:
        return None, None


def write_dataframe(df, file_name="abstract_results.csv"):
    data_path = f"{get_data_path()}/generated_answers/{file_name}"
    df.to_csv(data_path, index=False)
    print(f"Results saved to {data_path}")


def workflow(
    model: str = "gpt-4o-mini",
    dataset_name: str = "full-text.csv",
    temperature: float = 0.0,
    max_tokens: int = 500,
    use_abstract: bool = False,
    limit_papers: int = None,
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

    # get abstracts
    if use_abstract:
        dataset_path = "extracted_data/abstracts.xlsx"
        df = get_dataframe(file_name=dataset_path)
        col_name = "abstract"
    else:
        dataset_name = "full-text.csv"
        df = get_dataframe(file_name=dataset_name, read_csv=True)
        col_name = "text"

    # limit number of papers to process
    file_name = f"{col_name}_answers_{model}.csv"
    if limit_papers:
        df = df.sample(n=limit_papers, random_state=42)  # for reproducibility
        file_name = f"limit_n={limit_papers}_{file_name}"

    # convert column to list
    papers = df[col_name].tolist()  # list of full-text papers or abstracts
    titles = df["title"].tolist()  # list of titles

    # run workflow for each abstract
    data = []
    print(f"Running workflow for {len(papers)} papers...")
    for idx, (text, title) in enumerate(zip(papers, titles)):
        response = query_gpt(client, model, text, temperature, max_tokens)
        output, reasoning = extract_output_and_reasoning(response)
        data.append(
            {"title": title, "generated_answers": output, "reasoning": reasoning}
        )
        if idx % 1 == 0:
            print(f"processed {idx+1}/{len(papers)} abstracts...")

    # save results to a dataframe
    df_results = pd.DataFrame(data)
    write_dataframe(df_results, file_name)


def main(
    model: Annotated[
        str, typer.Option(help="The model to use for the analysis")
    ] = "gpt-4o-mini",
    dataset_name: Annotated[
        str, typer.Option(help="The name of the dataset")
    ] = "full-text.csv",
    temperature: Annotated[
        float, typer.Option(help="The temperature for sampling")
    ] = 0.0,
    max_tokens: Annotated[
        int, typer.Option(help="The maximum number of tokens to generate")
    ] = 500,
    use_abstract: Annotated[
        bool, typer.Option(help="Use a subset of the dataset")
    ] = False,
    limit_papers: Annotated[
        int, typer.Option(help="Limit the number of papers to process")
    ] = None,
):
    # run workflow
    workflow(
        model,
        dataset_name,
        temperature,
        max_tokens,
        use_abstract,
        limit_papers,
    )


# models: gpt-4o-mini, gpt-4o
# run this script in the terminal:
# gpt categorization generate_answers --model gpt-4o-mini --limit-papers 30 --use-subset
