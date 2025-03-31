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
                # "content": f"You are a helpful assistant designed to answer the user's prompt.",
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
def workflow(
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
    data_path = f"{get_data_path()}/{file_name}"
    df.to_csv(data_path, index=False)
    print(f"Results saved to {data_path}")


def main(
    model: Annotated[
        str, typer.Option(help="The model to use for the analysis")
    ] = "gpt-4o-mini",
    dataset_name: Annotated[
        str, typer.Option(help="The name of the dataset")
    ] = "abstracts.xlsx",
    temperature: Annotated[
        float, typer.Option(help="The temperature for sampling")
    ] = 0.0,
    max_tokens: Annotated[
        int, typer.Option(help="The maximum number of tokens to generate")
    ] = 500,
    use_subset: Annotated[
        bool, typer.Option(help="Use a subset of the dataset")
    ] = False,
):
    # load environment variables
    load_dotenv()

    # get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    # init OpenAI client
    client = OpenAI(api_key=api_key)

    # get abstracts
    df = get_dataframe(file_name=dataset_name)
    abstracts = df["abstract"].tolist()  # list of abstracts

    # parameters
    data = []
    file_name = f"output_results_{model}.csv"
    if use_subset:
        abstracts = abstracts[:5]  # use first 5 abstracts for testing
        file_name = f"test_output_results_{model}.csv"

    # run workflow for each abstract
    print(f"Running workflow for {len(abstracts)} abstracts...")
    for idx, abstract in enumerate(abstracts):
        response = workflow(client, model, abstract, temperature, max_tokens)
        output, reasoning = extract_output_and_reasoning(response)
        data.append({"abstract": abstract, "output": output, "reasoning": reasoning})
        if idx % 1 == 0:
            print(f"processed {idx+1}/{len(abstracts)} abstracts...")

    # save results to a dataframe
    df_results = pd.DataFrame(data)
    write_dataframe(df_results, file_name)


# models: gpt-4o-mini, gpt-4o
# run this script in the terminal:
# gpt categorization generate_answers --model gpt-4o-mini --use-subset --max-tokens 500
