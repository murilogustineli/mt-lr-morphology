import os
import re
import typer
from typing_extensions import Annotated
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from gpt.config import get_data_path, get_dataframe


def get_research_questions():
    research_questions = """\
    1. What challenges do varying degrees of morphological complexity (e.g., isolating, fusional, agglutinative, and polysynthetic languages) pose to machine translation systems in a low-resource language context?
    2. What techniques (e.g., rule-based methods, statistical models, or neural architectures) have been proposed to address these challenges?
    3. How do morphology-aware techniques (e.g. subword modeling, morphological analyzers) compare in effectiveness for low-resource machine translation?
    4. What are the specific findings, challenges, and proposed solutions and results for machine translation of languages in each different morphological typology (polysynthetic, agglutinative, fusional)?
    """
    return research_questions


# Chain-of-Thought prompt
def chain_of_thought_prompt(abstract: str, research_questions: str):
    prompt = f"""\
    You are an expert researcher. Your task is to analyze the abstract below and use its content to answer a set of specific research questions.

    ---

    Abstract:
    {abstract}

    ---

    Research Questions:
    {research_questions}

    ---

    Follow these steps carefully:

    Step 1: Read and Understand the Abstract
    Carefully read the abstract to extract all relevant information, including stated challenges, techniques used, comparisons, findings, and specific language typologies.

    Step 2: Analyze the Abstract for Each Research Question
    For each question below, extract answers **only if the abstract provides evidence or insights**. If the abstract does not include the required information, state that explicitly.

    Step 3: Provide a Structured Answer with Reasoning
    For each research question, write:
    - Answer: A direct response based on the abstract.
    - Evidence: A brief explanation or citation of the part of the abstract that supports your answer.

    ---

    Return your response in the following format:

    OUTPUT:
    1. Challenges of morphological complexity
    Answer:
    Evidence:

    2. Proposed techniques
    Answer:
    Evidence:

    3. Morphology-aware techniques**
    Answer:
    Evidence:

    4. Specific findings per morphological typology (e.g., polysynthetic, agglutinative, fusional)
    Answer:
    Evidence:

    ---

    REASONING: Briefly summarize what the abstract reveals overall in relation to the research questions. Note any gaps or limitations in the information provided.
    """
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
    ] = "dataset-updated.xlsx",
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
    file_name = f"output_results_{model}.csv"
    write_dataframe(df_results, file_name)
