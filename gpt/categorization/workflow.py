import os
import re
import typer
from typing_extensions import Annotated
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from gpt.config import get_data_path, get_dataframe


def get_inclusion_exclusion():
    inclusion_exclusion = """\
    1. Must focus on the machine translation task. We will exclude results that focus on other tasks, such as automatic speech recognition or named entity recognition.
    2. Must focus on at least one low resource language, which are languages with limited digital corpora.
    3. Must use morphology in its methodology or reference the effects of morphology in its presentation of the problem or its analysis of results.
    4. Must be peer-reviewed, must be a published academic conference paper, article, or book chapter.
    """
    return inclusion_exclusion


# Chain-of-Thought prompt
def get_prompt(abstract, inclusion_exclusion):
    prompt = f"""\
    I have an academic abstract that I would like to analyze based on specific inclusion and exclusion criteria.

    Step 1: Read the Abstract Carefully
    Abstract:
    {abstract}

    Step 2: Review the Inclusion and Exclusion Criteria
    {inclusion_exclusion}

    Step 3: Analyze the Abstract Step by Step
    - Does the abstract explicitly mention that it focuses on the machine translation task? Explain.
    - Does it mention at least one low-resource language? If so, which one(s)? Explain.
    - Does the methodology use morphology or discuss its effects? Provide details.
    - Is it a peer-reviewed academic paper (conference paper, article, or book chapter)? Provide reasoning.

    Step 4: Make a Final Decision
    Based on the above step-by-step analysis, classify the abstract into TRUE or FALSE and return the REASONING for the classification.

    Return the following:
    Step-by-Step Analysis and your reasoning for each step:
    - Mention of machine translation task: TRUE/FALSE
    - Mention of low-resource language(s): TRUE/FALSE
    - Use of morphology: TRUE/FALSE
    - Peer-reviewed status: TRUE/FALSE

    OUTPUT: TRUE or FALSE if this abstract meets the inclusion and exclusion criteria.
    REASONING: Provide a brief explanation summarizing your decision.
    """
    # Only return the OUTPUT and REASONING. Do not return any previous steps.
    return prompt


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
    prompt = get_prompt(abstract, get_inclusion_exclusion())
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
    temperature: Annotated[
        float, typer.Option(help="The temperature for sampling")
    ] = 0.0,
    max_tokens: Annotated[
        int, typer.Option(help="The maximum number of tokens to generate")
    ] = 250,
):
    # load environment variables
    load_dotenv()

    # get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")

    # init OpenAI client
    client = OpenAI(api_key=api_key)

    # get abstracts
    df = get_dataframe(file_name="dataset-updated.xlsx")
    abstracts = df["abstract"].tolist()  # list of 255 abstracts

    # run workflow for each abstract
    data = []
    for idx, abstract in enumerate(abstracts):
        response = workflow(client, model, abstract, temperature, max_tokens)
        output, reasoning = extract_output_and_reasoning(response)
        data.append({"abstract": abstract, "output": output, "reasoning": reasoning})
        if idx % 1 == 0:
            print(f"processed {idx+1}/{len(abstracts)} abstracts...")

    # save results to a dataframe
    df_results = pd.DataFrame(data)
    file_name = f"abstract_results_{model}_v2.csv"
    write_dataframe(df_results, file_name)
