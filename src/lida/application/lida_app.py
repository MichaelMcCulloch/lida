import logging
from typing import List, Union, Optional, Any, Dict
import pandas as pd
from llmx import TextGenerationConfig, TextGenerator

from lida.domain.models import Summary, Goal, Persona, ChartExecutorResponse
from lida.domain.ports import (
    ISummarizer,
    IGoalGenerator,
    IVizGenerator,
    IChartExecutor,
    IVizEditor,
    IVizExplainer,
    IVizEvaluator,
    IVizRepairer,
    IVizRecommender,
    IPersonaGenerator,
    IInfographicsGenerator,
)

logger = logging.getLogger("lida")


class LidaApp:
    def __init__(
        self,
        text_gen: TextGenerator,
        summarizer: ISummarizer,
        goal_generator: IGoalGenerator,
        viz_generator: IVizGenerator,
        chart_executor: IChartExecutor,
        viz_editor: Optional[IVizEditor] = None,
        viz_explainer: Optional[IVizExplainer] = None,
        viz_evaluator: Optional[IVizEvaluator] = None,
        viz_repairer: Optional[IVizRepairer] = None,
        viz_recommender: Optional[IVizRecommender] = None,
        persona_generator: Optional[IPersonaGenerator] = None,
        infographics_generator: Optional[IInfographicsGenerator] = None,
    ):
        self.text_gen = text_gen
        self.summarizer = summarizer
        self.goal_generator = goal_generator
        self.viz_generator = viz_generator
        self.chart_executor = chart_executor

        # Optional components
        self.viz_editor = viz_editor
        self.viz_explainer = viz_explainer
        self.viz_evaluator = viz_evaluator
        self.viz_repairer = viz_repairer
        self.viz_recommender = viz_recommender
        self.persona_generator = persona_generator
        self.infographics_generator = infographics_generator

        self.data = None

    def check_textgen(self, config: TextGenerationConfig):
        """
        Check if self.text_gen is the same as the config passed in. If not, update self.text_gen.
        """
        if config.provider is None:
            config.provider = self.text_gen.provider or "gemini"
            logger.info("Provider is not set, using default provider - %s", config.provider)
            return

        if self.text_gen.provider != config.provider:
            logger.info(
                "Switching Text Generator Provider from %s to %s",
                self.text_gen.provider,
                config.provider,
            )
            # This is a bit of a leakage, assuming we can just switch via llm() factory.
            # ideally the TextGenerator should be updated or a new one injected.
            from llmx import llm

            self.text_gen = llm(provider=config.provider)

    def summarize(
        self,
        data: Union[pd.DataFrame, str],
        file_name="",
        n_samples: int = 3,
        summary_method: str = "default",
        textgen_config: TextGenerationConfig = TextGenerationConfig(n=1, temperature=0),
    ) -> Summary:
        self.check_textgen(config=textgen_config)
        from lida.utils import read_dataframe

        # logic to handle file path specific to local execution context
        # Ideally this should be in the adapter or util, but LidaApp coordinating is fine.
        # But wait, read_dataframe reads from path.
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data)

        self.data = data
        return self.summarizer.summarize(
            data=self.data,
            text_gen=self.text_gen,
            file_name=file_name,
            n_samples=n_samples,
            summary_method=summary_method,
            textgen_config=textgen_config,
        )

    def goals(
        self,
        summary: Summary,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        n: int = 5,
        persona: Union[Persona, dict, str, None] = None,
    ) -> List[Goal]:
        self.check_textgen(config=textgen_config)

        if isinstance(persona, dict):
            persona = Persona(**persona)
        if isinstance(persona, str):
            persona = Persona(persona=persona, rationale="")

        return self.goal_generator.generate(
            summary=summary,
            text_gen=self.text_gen,
            textgen_config=textgen_config,
            n=n,
            persona=persona,
        )

    def visualize(
        self,
        summary: Summary,
        goal: Union[Goal, str, dict],
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library="seaborn",
        return_error: bool = False,
    ) -> List[ChartExecutorResponse]:
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str):
            goal = Goal(question=goal, visualization=goal, rationale="")

        self.check_textgen(config=textgen_config)

        code_specs = self.viz_generator.generate(
            summary=summary,
            goal=goal,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def execute(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        library: str = "seaborn",
        return_error: bool = False,
    ) -> List[ChartExecutorResponse]:
        import os
        import lida.web as lida
        from lida.utils import read_dataframe

        if data is None:
            # Fallback data loading logic
            root_file_path = os.path.dirname(os.path.abspath(lida.__file__))
            logger.info(f"Loading data from {summary.file_name}")
            data_path = os.path.join(root_file_path, "files/data", summary.file_name)
            if os.path.exists(data_path):
                data = read_dataframe(data_path)
            else:
                # What if data is not found?
                pass

        return self.chart_executor.execute(
            code_specs=code_specs,
            data=data,
            summary=summary,
            library=library,
            return_error=return_error,
        )

    def edit(
        self,
        code: str,
        summary: Summary,
        instructions: Union[str, List[str]],
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ) -> List[ChartExecutorResponse]:
        """Edit a visualization code given a set of instructions"""
        if not self.viz_editor:
            raise ValueError("VizEditor adapter not initialized")

        self.check_textgen(config=textgen_config)

        if isinstance(instructions, str):
            instructions = [instructions]

        code_specs = self.viz_editor.generate(
            code=code,
            summary=summary,
            instructions=instructions,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def repair(
        self,
        code: str,
        goal: Goal,
        summary: Summary,
        feedback: Union[str, Dict, List[Dict]],
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ) -> List[ChartExecutorResponse]:
        """Repair a visulization given some feedback"""
        if not self.viz_repairer:
            raise ValueError("VizRepairer adapter not initialized")

        self.check_textgen(config=textgen_config)

        code_specs = self.viz_repairer.generate(
            code=code,
            feedback=feedback,
            goal=goal,
            summary=summary,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def explain(
        self,
        code: str,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ) -> List[Dict]:
        """Explain a visualization code"""
        if not self.viz_explainer:
            raise ValueError("VizExplainer adapter not initialized")

        self.check_textgen(config=textgen_config)
        return self.viz_explainer.generate(
            code=code,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def evaluate(
        self,
        code: str,
        goal: Goal,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
    ) -> List[Dict]:
        """Evaluate a visualization code given a goal"""
        if not self.viz_evaluator:
            raise ValueError("VizEvaluator adapter not initialized")

        self.check_textgen(config=textgen_config)

        return self.viz_evaluator.generate(
            code=code,
            goal=goal,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )

    def recommend(
        self,
        code: str,
        summary: Summary,
        n: int = 4,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        library: str = "seaborn",
        return_error: bool = False,
    ) -> List[ChartExecutorResponse]:
        """Recommend visualization code"""
        if not self.viz_recommender:
            raise ValueError("VizRecommender adapter not initialized")

        self.check_textgen(config=textgen_config)

        code_specs = self.viz_recommender.generate(
            code=code,
            summary=summary,
            n=n,
            textgen_config=textgen_config,
            text_gen=self.text_gen,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

    def personas(
        self,
        summary: Summary,
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        n: int = 5,
    ) -> List[Persona]:
        """Generate personas given a summary"""
        if not self.persona_generator:
            raise ValueError("PersonaGenerator adapter not initialized")

        self.check_textgen(config=textgen_config)
        return self.persona_generator.generate(
            summary=summary, text_gen=self.text_gen, textgen_config=textgen_config, n=n
        )

    def infographics(
        self,
        visualization: str,
        n: int = 1,
        style_prompt: Union[str, List[str]] = "",
        return_pil: bool = False,
    ) -> Any:
        """Generate infographics"""
        if not self.infographics_generator:
            # Try to lazy load or error?
            # For DDD, we should have injected it.
            # If it's optional, we raise or info user.
            # The Manager handled the import error.
            # LidaApp should just assume it's supplied or fail.
            raise ValueError(
                "InfographicsGenerator adapter not initialized. Please ensure lida[infographics] is installed and configured."
            )

        # No textgen check needed for image gen usually, unless it uses prompts?
        # The style_prompt is passed.

        return self.infographics_generator.generate(
            visualization=visualization,
            n=n,
            style_prompt=style_prompt,
            return_pil=return_pil,
        )
