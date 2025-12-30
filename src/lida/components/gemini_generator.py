import os
import time
from google import genai
from google.genai import types
from llmx.generators.text.base_textgen import TextGenerator
from llmx.datamodel import TextGenerationConfig, TextGenerationResponse, Message
import logging

logger = logging.getLogger("lida")


class GeminiTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = None,
        provider: str = "google",
        model: str = None,
    ):
        super().__init__(provider=provider)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set for Gemini provider.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model or "gemini-3-flash-preview"

    def generate(
        self,
        messages,
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        model_name = config.model or self.model_name

        # Prepare content for generate_content
        # google-genai expects 'contents' which can be a string or list of Content objects
        # We need to map role 'user'/'model' correctly.

        contents = []
        system_instruction = None

        if isinstance(messages, list):
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")

                if role == "system":
                    system_instruction = content
                elif role == "user":
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
                elif role == "assistant" or role == "model":
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

        # Configure generation config
        gen_config = types.GenerateContentConfig(
            candidate_count=config.n,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            system_instruction=system_instruction,
        )

        try:
            # According to new SDK, we use client.models.generate_content
            print(
                f"DEBUG: Calling Gemini generate_content with model {model_name}...",
                flush=True,
            )
            start_time = time.time()
            response = self.client.models.generate_content(model=model_name, contents=contents, config=gen_config)
            end_time = time.time()
            duration = end_time - start_time
            print(
                f"DEBUG: Gemini generation completed in {duration:.2f}s using model {model_name}",
                flush=True,
            )

            # Map response to TextGenerationResponse
            candidates = []

            # response.candidates is a list
            if response.candidates:
                for cand in response.candidates:
                    # content parts
                    if cand.content and cand.content.parts:
                        text_content = "".join([part.text for part in cand.content.parts if part.text])
                        candidates.append(Message(role="assistant", content=text_content))

            # Fallback if text helper is available and candidates are empty?
            if not candidates and hasattr(response, "text"):
                candidates.append(Message(role="assistant", content=response.text))

            return TextGenerationResponse(text=candidates, logprobs=[], config=config, usage={"total_tokens": 0})

        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            raise e

    def count_tokens(self, text) -> int:
        return 0
