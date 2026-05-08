import json
import logging
import warnings
from typing import Dict, List, Optional, Union
import pandas as pd
from lida.llm_types import TextGenerator, TextGenerationConfig

from lida.domain.models import Summary
from lida.domain.ports import ISummarizer
from lida.utils import adapt_messages_for_provider, clean_code_snippet, read_dataframe


logger = logging.getLogger("lida")

SYSTEM_PROMPT = """
You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description
ii) ALWAYS generate a field description.
iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
You must return an updated JSON dictionary without any preamble or explanation.
"""


class SummarizerAdapter(ISummarizer):
    def __init__(self, text_gen: TextGenerator):
        self.text_gen = text_gen

    def check_type(self, dtype: str, value):
        """Cast value to right type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
        """Get properties of each column in a pandas DataFrame"""
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                properties["std"] = self.check_type(dtype, df[column].std())
                properties["min"] = self.check_type(dtype, df[column].min())
                properties["max"] = self.check_type(dtype, df[column].max())

            elif pd.api.types.is_bool_dtype(df[column]):
                properties["dtype"] = "boolean"
            elif pd.api.types.is_object_dtype(df[column]):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors="raise")
                        properties["dtype"] = "date"
                except ValueError:
                    if df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            if properties["dtype"] == "date":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors="coerce")
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()

            nunique = df[column].nunique()
            if "samples" not in properties:
                non_null_values = df[column][df[column].notnull()].unique()
                n_samples = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            properties["semantic_type"] = ""
            properties["description"] = ""
            properties_list.append({"column": column, "properties": properties})

        return properties_list

    def enrich(
        self,
        base_summary: dict,
        textgen_config: TextGenerationConfig,
    ) -> dict:
        """Enrich the data summary with descriptions"""
        logger.info("Enriching the data summary with descriptions")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "assistant",
                "content": f"""
        Annotate the dictionary below. Only return a JSON object.
        {base_summary}
        """,
            },
        ]

        messages = adapt_messages_for_provider(messages, self.text_gen.provider)
        response = self.text_gen.generate(messages=messages, config=textgen_config)
        enriched_summary = base_summary
        try:
            json_string = clean_code_snippet(response.text[0]["content"])
            enriched_summary = json.loads(json_string)
        except json.decoder.JSONDecodeError:
            error_content = response.text[0]["content"]
            error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Consider using a default summary or  a larger model with higher max token length. | {error_content}"
            logger.error(error_msg)
            logger.debug(f"LLM Usage: {response.usage}")
            raise ValueError(error_msg)
        return enriched_summary

    def summarize(
        self,
        data: Union[pd.DataFrame, str],
        summary_method: str = "default",
        textgen_config: Optional[TextGenerationConfig] = None,
    ) -> Summary:
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data, encoding="utf-8")
        else:
            file_name = ""

        if textgen_config is None:
            textgen_config = TextGenerationConfig(n=1)

        data_properties = self.get_column_properties(data)

        base_summary = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
        }

        data_summary = base_summary

        if summary_method == "llm":
            data_summary = self.enrich(base_summary, textgen_config=textgen_config)
        elif summary_method == "columns":
            data_summary = {
                "name": file_name,
                "file_name": file_name,
                "dataset_description": "",
                "fields": data_properties,
            }

        data_summary["field_names"] = data.columns.tolist()
        data_summary["file_name"] = file_name

        # Map to Domain Model
        return Summary(
            name=data_summary.get("name", ""),
            file_name=data_summary.get("file_name", ""),
            dataset_description=data_summary.get("dataset_description", ""),
            field_names=data_summary.get("field_names", []),
            fields=data_summary.get("fields", []),
        )
