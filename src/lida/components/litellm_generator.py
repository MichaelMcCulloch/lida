import logging
import os
import time
from typing import Dict, List, Optional

from openai import OpenAI

from lida.llm_types import Message, TextGenerationConfig, TextGenerationResponse, TextGenerator

logger = logging.getLogger("lida")

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "http://localhost:4000"


class LiteLLMTextGenerator(TextGenerator):
    """OpenAI-format chat completions client pointed at a LiteLLM proxy.

    Endpoint and token are read from env: LITELLM_BASE_URL, LITELLM_API_KEY.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "litellm",
    ):
        super().__init__(provider=provider)
        self.base_url = base_url or os.environ.get("LITELLM_BASE_URL") or DEFAULT_BASE_URL
        # LiteLLM proxies typically require a token, but allow a sentinel so the
        # OpenAI client doesn't error out when running against an open proxy.
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY") or "sk-litellm-noop"
        self.model_name = model or os.environ.get("LITELLM_MODEL") or DEFAULT_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(
        self,
        messages: List[Dict[str, str]],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        model_name = config.model or self.model_name

        request: Dict = {
            "model": model_name,
            "messages": messages,
            "n": config.n,
            "temperature": config.temperature,
        }
        if config.max_tokens:
            request["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.stop:
            request["stop"] = config.stop

        start = time.time()
        response = self.client.chat.completions.create(**request)
        logger.info("LiteLLM call (model=%s) completed in %.2fs", model_name, time.time() - start)

        text = [Message(role=choice.message.role or "assistant", content=choice.message.content or "") for choice in response.choices]

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return TextGenerationResponse(text=text, logprobs=[], config=config, usage=usage)

    def count_tokens(self, text) -> int:
        return 0
