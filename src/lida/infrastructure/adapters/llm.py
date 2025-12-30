import json
import logging
from typing import Dict, List, Optional, Union

from lida.utils import clean_code_snippet, adapt_messages_for_provider
from llmx import TextGenerator, TextGenerationConfig, TextGenerationResponse
from lida.domain.models import Goal, Persona, Summary
from lida.domain.ports import (
    IGoalGenerator,
    IVizGenerator,
    IVizEditor,
    IVizExplainer,
    IVizEvaluator,
    IVizRepairer,
    IVizRecommender,
    IPersonaGenerator,
)
from lida.infrastructure.services.scaffold import ChartScaffold

logger = logging.getLogger("lida")

SYSTEM_PROMPT_PERSONA = """You are an experienced data analyst  who can take a dataset summary and generate a list of n personas (e.g., ceo or accountant for finance related data, economist for population or gdp related data, doctors for health data, or just users) that might be critical stakeholders in exploring some data and describe rationale for why they are critical. The personas should be prioritized based on their relevance to the data. Think step by step.

Your response should be perfect JSON in the following format:
```[{"persona": "persona1", "rationale": "..."},{"persona": "persona1", "rationale": "..."}]```
"""

SYSTEM_INSTRUCTIONS_GOAL = """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
"""

FORMAT_INSTRUCTIONS_GOAL = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

SYSTEM_PROMPT_VIZ = """
You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Given some code template, you complete the template to generate a visualization given the dataset and the goal described. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc. .
"""

SYSTEM_PROMPT_EDITOR = """
You are a high skilled visualization assistant that can modify a provided visualization code based on a set of instructions. You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""

SYSTEM_PROMPT_REPAIRER = """
You are a helpful assistant highly skilled in revising visualization code to improve the quality of the code and visualization based on feedback.  Assume that data in plot(data) contains a valid dataframe.
You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""

SYSTEM_PROMPT_EXPLAINER = """
You are a helpful assistant highly skilled in providing helpful, structured explanations of visualization of the plot(data: pd.DataFrame) method in the provided code. You divide the code into sections and provide a description of each section and an explanation. The first section should be named "accessibility" and describe the physical appearance of the chart (colors, chart type etc), the goal of the chart, as well the main insights from the chart.
You can explain code across the following 3 dimensions:
1. accessibility: the physical appearance of the chart (colors, chart type etc), the goal of the chart, as well the main insights from the chart.
2. transformation: This should describe the section of the code that applies any kind of data transformation (filtering, aggregation, grouping, null value handling etc)
3. visualization: step by step description of the code that creates or modifies the presented visualization.

"""

FORMAT_INSTRUCTIONS_EXPLAINER = """
Your output MUST be perfect JSON in THE FORM OF A VALID LIST of JSON OBJECTS WITH PROPERLY ESCAPED SPECIAL CHARACTERS e.g.,

```[
    {"section": "accessibility", "code": "None", "explanation": ".."}  , {"section": "transformation", "code": "..", "explanation": ".."}  ,  {"section": "visualization", "code": "..", "explanation": ".."}
    ] ```

The code part of the dictionary must come from the supplied code and should cover the explanation. The explanation part of the dictionary must be a string. The section part of the dictionary must be one of "accessibility", "transformation", "visualization" with no repetition. THE LIST MUST HAVE EXACTLY 3 JSON OBJECTS [{}, {}, {}].  THE GENERATED JSON  MUST BE A LIST IE START AND END WITH A SQUARE BRACKET.
"""

SYSTEM_PROMPT_EVALUATOR = """
You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
- bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
- Data transformation (transformation): Is the data transformed appropriately for the visualization type? E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed?
- Goal compliance (compliance): how well the code meets the specified visualization goals?
- Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST be less than 5.
- Data encoding (encoding): Is the data encoded appropriately for the visualization type?
- aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data?

You must provide a score for each of the above dimensions.  Assume that data in chart = plot(data) contains a valid dataframe for the dataset. The `plot` function returns a chart (e.g., matplotlib, seaborn etc object).

Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:

```[
{ "dimension":  "bugs",  "score": x , "rationale": " .."}, { "dimension":  "transformation",  "score": x, "rationale": " .."}, { "dimension":  "compliance",  "score": x, "rationale": " .."},{ "dimension":  "type",  "score": x, "rationale": " .."}, { "dimension":  "encoding",  "score": x, "rationale": " .."}, { "dimension":  "aesthetics",  "score": x, "rationale": " .."}
]
```
"""

SYSTEM_PROMPT_RECOMMENDER = """

You are a helpful assistant highly skilled in recommending a DIVERSE set of visualization code. Your input is an example visualization code,  a summary of a dataset and an example visualization goal that the user has already seen. Given this input, your task is to recommend additional visualizations that a user may be interested. Your recommendation may consider different types of valid data aggregations, chart types, clearer ways of displaying information and uses different variables from the data summary. THE CODE YOU GENERATE MUST BE CORRECT (follow the language syntax and syntax of the visualization grammar) AND FOLLOW VISUALIZATION BEST PRACTICES.

Your output MUST be a n code snippets separated by ******* (5 asterisks). Each snippet MUST BE AN independent code snippet (with one plot method) similar to the example code. For example

```python
# code snippet 1
import  ...
....
```
*****

```python
# code snippet 2
import ...
....
```

```python
# code snippet n
import ...
....
```


"""


class GoalGenerator(IGoalGenerator):
    """Generate goals given a summary of data"""

    def generate(
        self,
        summary: Summary,
        text_gen: TextGenerator,
        textgen_config: TextGenerationConfig,
        n=5,
        persona: Optional[Persona] = None,
    ) -> List[Goal]:
        """Generate goals given a summary of data"""

        summary_dict = summary.__dict__

        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n .
        {summary_dict} \n\n"""

        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="",
            )

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona} persona, who is insterested in complex, insightful goals about the data. \n"""

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS_GOAL},
            {
                "role": "assistant",
                "content": f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS_GOAL} \n\n. The generated {n} goals are: \n ",
            },
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        result = text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result_json = json.loads(json_string)
            if isinstance(result_json, dict):
                result_json = [result_json]
            goals = [Goal(**x) for x in result_json]
        except json.decoder.JSONDecodeError:
            error_content = result.text[0]["content"]
            logger.error(f"Error decoding JSON from LLM response: {error_content}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate goals. Consider using a larger model or a model with higher max token length."
            )
        return goals


class VizGenerator(IVizGenerator):
    """Generate visualizations from prompt"""

    def __init__(self) -> None:
        self.scaffold = ChartScaffold()

    def generate(
        self,
        summary: Summary,
        goal: Goal,
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        library="altair",
    ) -> List[str]:
        """Generate visualization code given a summary and a goal"""

        summary_dict = summary.__dict__

        library_template, library_instructions = self.scaffold.get_template(goal, library)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_VIZ},
            {
                "role": "system",
                "content": f"The dataset summary is : {summary_dict} \n\n",
            },
            library_instructions,
            {
                "role": "user",
                "content": f"Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n",
            },
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        completions: TextGenerationResponse = text_gen.generate(messages=messages, config=textgen_config)
        response = [x["content"] for x in completions.text]

        return response


class VizEditor(IVizEditor):
    """Generate visualizations from prompt"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
        self,
        code: str,
        summary: Summary,
        instructions: list[str],
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        library="altair",
    ):
        """Edit a code spec based on instructions"""

        instruction_string = ""
        for i, instruction in enumerate(instructions):
            instruction_string += f"{i + 1}. {instruction} \n"

        library_template, library_instructions = self.scaffold.get_template(
            Goal(index=0, question="", visualization="", rationale=""), library
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EDITOR},
            {
                "role": "system",
                "content": f"The dataset summary is : \n\n {summary} \n\n",
            },
            {
                "role": "system",
                "content": f"The modifications you make MUST BE CORRECT and  based on the '{library}' library and also follow these instructions \n\n{library_instructions} \n\n. The resulting code MUST use the following template \n\n {library_template} \n\n ",
            },
            {
                "role": "user",
                "content": f"ALL ADDITIONAL LIBRARIES USED MUST BE IMPORTED.\n The code to be modified is: \n\n{code} \n\n. YOU MUST THINK STEP BY STEP, AND CAREFULLY MODIFY ONLY the content of the plot(..) method TO MEET EACH OF THE FOLLOWING INSTRUCTIONS: \n\n {instruction_string} \n\n. The completed modified code THAT FOLLOWS THE TEMPLATE above is. \n",
            },
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        completions: TextGenerationResponse = text_gen.generate(messages=messages, config=textgen_config)
        return [x["content"] for x in completions.text]


class VizRepairer(IVizRepairer):
    """Fix visualization code based on feedback"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
        self,
        code: str,
        feedback: Union[str, Dict, List[Dict]],
        goal: Goal,
        summary: Summary,
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        library="altair",
    ):
        """Fix a code spec based on feedback"""
        library_template, library_instructions = self.scaffold.get_template(
            Goal(index=0, question="", visualization="", rationale=""), library
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_REPAIRER},
            {
                "role": "system",
                "content": f"The dataset summary is : {summary}. \n . The original goal was: {goal}.",
            },
            {
                "role": "system",
                "content": f"You MUST use only the {library}. The resulting code MUST use the following template {library_template}. Only use variables that have been defined in the code or are in the dataset summary",
            },
            {
                "role": "user",
                "content": f"The existing code to be fixed is: {code}. \n Fix the code above to address the feedback: {feedback}. ONLY apply feedback that are CORRECT.",
            },
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        completions: TextGenerationResponse = text_gen.generate(messages=messages, config=textgen_config)
        return [x["content"] for x in completions.text]


class VizExplainer(IVizExplainer):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
        self,
        code: str,
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        library="seaborn",
    ):
        """Generate a visualization explanation given some code"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EXPLAINER},
            {
                "role": "assistant",
                "content": f"The code to be explained is {code}.\n=======\n",
            },
            {
                "role": "user",
                "content": f"{FORMAT_INSTRUCTIONS_EXPLAINER}. \n\n. The structured explanation for the code above is \n\n",
            },
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        completions: TextGenerationResponse = text_gen.generate(messages=messages, config=textgen_config)

        completions = [clean_code_snippet(x["content"]) for x in completions.text]
        explanations = []

        for completion in completions:
            try:
                exp = json.loads(completion)
                explanations.append(exp)
            except Exception as e:
                print("Error parsing completion", completion, str(e))
        return explanations


class VizEvaluator(IVizEvaluator):
    """Generate visualizations Explanations given some code"""

    def __init__(
        self,
    ) -> None:
        pass

    def generate(
        self,
        code: str,
        goal: Goal,
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        library="altair",
    ):
        """Generate a visualization explanation given some code"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EVALUATOR},
            {
                "role": "assistant",
                "content": f"Generate an evaluation given the goal and code below in {library}. The specified goal is \n\n {goal.question} \n\n and the visualization code is \n\n {code} \n\n. Now, evaluate the code based on the 6 dimensions above. \n. THE SCORE YOU ASSIGN MUST BE MEANINGFUL AND BACKED BY CLEAR RATIONALE. A SCORE OF 1 IS POOR AND A SCORE OF 10 IS VERY GOOD. The structured evaluation is below .",
            },
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        completions: TextGenerationResponse = text_gen.generate(messages=messages, config=textgen_config)

        completions = [clean_code_snippet(x["content"]) for x in completions.text]
        evaluations = []
        for completion in completions:
            try:
                evaluation = json.loads(completion)
                evaluations.append(evaluation)
            except Exception as json_error:
                print("Error parsing evaluation data", completion, str(json_error))
        return evaluations


class VizRecommender(IVizRecommender):
    """Generate visualizations from prompt"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
        self,
        code: str,
        summary: Summary,
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        n=3,
        library="seaborn",
    ):
        """Recommend a code spec based on existing visualization"""

        library_template, library_instructions = self.scaffold.get_template(
            Goal(index=0, question="", visualization="", rationale=""), library
        )

        structure_instruction = f"""
        EACH CODE SNIPPET MUST BE A FULL PROGRAM (IT MUST IMPORT ALL THE LIBRARIES THAT ARE USED AND MUST CONTAIN plot(data) method). IT MUST FOLLOW THE STRUCTURE BELOW AND ONLY MODIFY THE INDICATED SECTIONS. \n\n {library_template} \n\n.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RECOMMENDER},
            {"role": "system", "content": structure_instruction},
            {
                "role": "system",
                "content": f"The dataset summary is : \n\n {summary} \n\n",
            },
            {
                "role": "system",
                "content": f"An example visualization code is: \n\n ```{code}``` \n\n. You MUST use only the {library} library. \n",
            },
            {
                "role": "user",
                "content": f"Recommend {n} (n=({n})) visualizations in the format specified. \n.",
            },
        ]
        messages = adapt_messages_for_provider(messages, text_gen.provider)
        textgen_config.messages = messages
        result: TextGenerationResponse = text_gen.generate(messages=messages, config=textgen_config)
        output = []
        try:
            snippets = result.text[0]["content"].split("*****")
            for snippet in snippets:
                cleaned_snippet = clean_code_snippet(snippet)
                if len(cleaned_snippet) > 4:
                    output.append(cleaned_snippet)
        except Exception:
            # Fallback if split fails or content is not as expected
            pass
        return output


class PersonaGenerator(IPersonaGenerator):
    """Generate personas given a summary of data"""

    def generate(
        self,
        summary: Summary,
        text_gen: TextGenerator,
        textgen_config: TextGenerationConfig,
        n=5,
    ) -> List[Persona]:
        """Generate personas given a summary of data"""

        summary_dict = summary.__dict__

        user_prompt = (
            f"The number of PERSONAs to generate is {n}. Generate {n} personas in the right format given the data summary below,\n .\n"
            f"{summary_dict} \n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PERSONA},
            {"role": "assistant", "content": user_prompt},
        ]

        messages = adapt_messages_for_provider(messages, text_gen.provider)
        result = text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result_json = json.loads(json_string)
            if isinstance(result_json, dict):
                result_json = [result_json]
            personas = [Persona(**x) for x in result_json]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate personas. Consider using a larger model."
            )
        return personas
