import base64
import logging
from typing import List, Tuple, Union, Dict
import os
import io
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import tiktoken

logger = logging.getLogger("lida")


def adapt_messages_for_provider(messages: List[Dict[str, str]], provider: str) -> List[Dict[str, str]]:
    """Adapts the message structure for a specific LLM provider."""
    if provider == "gemini" or provider == "google":
        return _adapt_for_gemini(messages)
    return messages


def get_provider_preference() -> str:
    """
    Determine local preference for LLM provider based on available keys.
    Preference: Gemini (google) > OpenAI > Anthropic
    """
    # Ensure GEMINI_API_KEY is mapped to GOOGLE_API_KEY if needed by libraries
    google_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if google_key:
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = google_key
        # llmx's google/palm provider often requires PALM_API_KEY
        if not os.environ.get("PALM_API_KEY"):
            os.environ["PALM_API_KEY"] = google_key
        return "google"

    elif os.environ.get("OPENAI_API_KEY"):
        return "openai"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"  # Default fallback


def _adapt_for_gemini(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Converts a list of messages to Gemini's required format:
    - No system prompts.
    - Roles must be 'user' and 'model'.
    - Must start with a 'user' role.
    - Roles must alternate.
    """
    # 1. Consolidate all content into a list of (role, content) tuples, mapping roles.
    content_tuples = []
    system_content = []
    for msg in messages:
        if not msg.get("content") or not msg.get("content").strip():
            continue  # Skip messages with no content
        if msg["role"] == "system":
            system_content.append(msg["content"])
        elif msg["role"] == "user":
            content_tuples.append(("user", msg["content"]))
        elif msg["role"] == "assistant":
            content_tuples.append(("model", msg["content"]))

    # 2. Prepend system prompt to the first content item.
    if system_content:
        full_system_prompt = "\n\n".join(system_content)
        if content_tuples:
            first_role, first_content = content_tuples[0]
            content_tuples[0] = (first_role, f"{full_system_prompt}\n\n{first_content}")
        else:
            content_tuples.append(("user", full_system_prompt))

    if not content_tuples:
        return []

    # 3. Merge consecutive messages of the same role.
    merged_tuples = []
    for role, content in content_tuples:
        if merged_tuples and merged_tuples[-1][0] == role:
            merged_tuples[-1] = (role, f"{merged_tuples[-1][1]}\n\n{content}")
        else:
            merged_tuples.append((role, content))

    # 4. Ensure the conversation starts with a 'user' role.
    if merged_tuples and merged_tuples[0][0] == "model":
        first_model_content = merged_tuples.pop(0)[1]
        if merged_tuples and merged_tuples[0][0] == "user":
            merged_tuples[0] = (
                "user",
                f"{first_model_content}\n\n{merged_tuples[0][1]}",
            )
        else:
            merged_tuples.insert(0, ("user", first_model_content))

    # 5. Build the final list of dicts.
    adapted_messages = [{"role": role, "content": content} for role, content in merged_tuples]

    return adapted_messages


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def clean_column_name(col_name: str) -> str:
    """
    Clean a single column name by replacing special characters and spaces with underscores.

    :param col_name: The name of the column to be cleaned.
    :return: A sanitized string valid as a column name.
    """
    return re.sub(r"[^0-9a-zA-Z_]", "_", col_name)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all column names in the given DataFrame.

    :param df: The DataFrame with possibly dirty column names.
    :return: A copy of the DataFrame with clean column names.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = [clean_column_name(col) for col in cleaned_df.columns]
    return cleaned_df


def read_dataframe(file_location: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Read a dataframe from a given file location.
    It samples down to 4500 rows if the data exceeds that limit and cleans column names.

    :param file_location: The path to the file containing the data.
    :param encoding: Encoding to use for the file reading.
    :return: A cleaned and sampled DataFrame.
    """
    file_extension = file_location.split(".")[-1]

    # Handle GZIP compressed files
    if file_extension == "gz":
        file_extension = file_location.split(".")[-2]

    read_funcs = {
        "json": lambda: pd.read_json(file_location, orient="records", encoding=encoding, compression="infer"),
        "csv": lambda: pd.read_csv(file_location, encoding=encoding, compression="infer"),
        "xls": lambda: pd.read_excel(file_location),
        "xlsx": lambda: pd.read_excel(file_location),
        "parquet": pd.read_parquet,
        "feather": pd.read_feather,
        "tsv": lambda: pd.read_csv(file_location, sep="	", encoding=encoding, compression="infer"),
    }

    if file_extension not in read_funcs:
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        df = read_funcs[file_extension]()
    except Exception as e:
        logger.error(f"Failed to read file: {file_location}. Error: {e}")
        raise

    # Clean column names
    df = clean_column_names(df)

    # Sample down to 4500 rows if necessary
    if len(df) > 4500:
        logger.info(f"Dataframe has more than 4500 rows ({len(df)}). Sampling down to 4500 rows.")
        df = df.sample(4500, random_state=42)

    return df


def file_to_df(file_location: str):
    """Get summary of data from file location"""
    file_name = file_location.split("/")[-1]
    df = None
    if "csv" in file_name:
        df = pd.read_csv(file_location)
    elif "xlsx" in file_name:
        df = pd.read_excel(file_location)
    elif "json" in file_name:
        df = pd.read_json(file_location, orient="records")
    elif "parquet" in file_name:
        df = pd.read_parquet(file_location)
    elif "feather" in file_name:
        df = pd.read_feather(file_location)

    return df


def plot_raster(rasters: Union[str, List[str]], figsize: Tuple[int, int] = (10, 10)):
    """
    Plot a series of base64-encoded raster images in a horizontal layout.

    Args:
        rasters: A single base64 string or a list of base64-encoded strings representing the images.
        figsize: A tuple indicating the size of the figure to display.
    """
    plt.figure(figsize=figsize)

    if isinstance(rasters, str):
        rasters = [rasters]

    images = []

    # Find the max height for resizing
    max_height = 0
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image = plt.imread(io.BytesIO(decoded_image), format="PNG")

        max_height = max(max_height, image.shape[0])

    # Resize images to max_height while preserving the aspect ratio and alpha channel if it exists
    for raster in rasters:
        decoded_image = base64.b64decode(raster)
        image = plt.imread(io.BytesIO(decoded_image), format="PNG")

        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = int(max_height * aspect_ratio)
        image_resized = np.array(
            [np.interp(np.linspace(0, len(row), new_width), np.arange(0, len(row)), row) for row in image]
        )

        if image_resized.shape[2] == 4:  # If RGBA, preserve alpha channel
            alpha_channel = image_resized[:, :, 3:]
            # Drop the alpha for visualization
            image_resized = image_resized[:, :, :3]
            image_resized = np.clip(image_resized, 0, 1)
            image_resized = np.concatenate((image_resized, alpha_channel), axis=2)

        images.append(image_resized)

    # Concatenate images along the width
    concatenated_image = np.concatenate(images, axis=1)

    plt.imshow(concatenated_image)
    plt.axis("off")
    plt.show()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")





def clean_code_snippet(code_string):
    # Extract code snippet using regex
    cleaned_snippet = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", code_string)

    if cleaned_snippet:
        cleaned_snippet = cleaned_snippet.group(1)
    else:
        cleaned_snippet = code_string

    # remove non-printable characters
    # cleaned_snippet = re.sub(r'[ - ]+', ' ', cleaned_snippet)

    return cleaned_snippet
