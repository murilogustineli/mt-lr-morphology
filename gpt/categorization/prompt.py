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
