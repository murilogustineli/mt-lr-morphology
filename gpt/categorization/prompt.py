def get_research_questions():
    research_questions = """\
    1. Does the paper discuss the challenges posed by the morphological complexity of the target language(s)? If so, what specific challenges are identified for languages with isolating, fusional, agglutinative, or polysynthetic structures in a low-resource context?
    2. What techniques does the paper propose or evaluate to address the challenges of morphological complexity in low-resource machine translation? For instance, are these rule-based, statistical, or neural methods?
    3. If there are any morphology-aware techniques used in the paper (e.g., subword modeling, morphological analyzers, morpheme segmentation), how do they compare to other approaches or baselines in terms of effectiveness?
    4. Does the paper provide findings, identify challenges, or propose solutions that are specific to any particular morphological typology (e.g., polysynthetic, agglutinative, fusional)? What are the reported outcomes for each typology, if any?
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

    3. Morphology-aware techniques
    Answer:
    Evidence:

    4. Specific findings per morphological typology (e.g., polysynthetic, agglutinative, fusional)
    Answer:
    Evidence:

    ---

    REASONING: Briefly summarize what the abstract reveals overall in relation to the research questions. Note any gaps or limitations in the information provided.
    """
    return prompt
