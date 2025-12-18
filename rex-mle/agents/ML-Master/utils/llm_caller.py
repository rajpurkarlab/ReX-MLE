import importlib
import os
import time
import logging
from typing import List, Dict, Union, Optional, Any

from openai import OpenAI
from anthropic import Anthropic
logger = logging.getLogger("ml-master")


class LLM:
    """
    Encapsulate the VLLM-based LLM class to invoke the self-hosted VLLM model via the OpenAI SDK.
    """
    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure every message.content is either a string or a list of text parts.

        Converts a single content dict with type/text into a one-element list so that
        the API always receives the expected shape.
        """
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError(f"Each message must be a dict, got {type(msg)}")
            content = msg.get("content")
            if isinstance(content, str):
                normalized_content = content
            elif isinstance(content, list):
                normalized_content = []
                for item in content:
                    if isinstance(item, dict) and "type" in item and "text" in item and isinstance(item["text"], str):
                        normalized_content.append({"type": item["type"], "text": item["text"]})
                    else:
                        raise ValueError(f"Invalid message content item: {item}")
            elif isinstance(content, dict) and "type" in content and "text" in content and isinstance(content["text"], str):
                normalized_content = [{"type": content["type"], "text": content["text"]}]
            elif isinstance(content, dict) and "type" in content and "text" in content and isinstance(content["text"], list):
                normalized_content = [{"type": content["type"], "text": text} for text in content["text"]]
            else:
                raise ValueError(
                    "Invalid message content: expected a string or list of {type, text} dicts"
                )

            normalized.append({**msg, "content": normalized_content})
        return normalized
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",
        model_name: str = "default-model",
        temperature: float = 0.7,
        max_tokens: int = 16384,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        retry_time: int = 20,
        delay_time: int = 3,
    ):
        """
        Initialize the VLLM LLM class.

        Args:
            base_url: The URL of the VLLM service or Azure endpoint.
            api_key: API key.
            model_name: Name of the model/deployment.
            temperature: Temperature parameter to control output randomness.
            max_tokens: Maximum number of tokens to generate.
            stop_tokens: List of tokens that signal the end of generation.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens
        self.retry_time = retry_time
        self.delay_time = delay_time
        self.provider = self._determine_provider(model_name)

        if self.provider == "openai":
            # Check if using Azure OpenAI (has /openai/deployments/ in URL)
            is_azure = "/openai/deployments/" in base_url
            if is_azure:
                # For Azure, extract deployment name and API version from base_url
                # We need base_url without /chat/completions and api-version as query param
                if "?api-version=" in base_url:
                    base_without_version, version_part = base_url.split("?api-version=")
                    api_version = version_part
                    # Remove /chat/completions if present
                    if base_without_version.endswith("/chat/completions"):
                        base_without_version = base_without_version[:-len("/chat/completions")]

                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url=base_without_version,
                        default_query={"api-version": api_version},
                        default_headers={"api-key": self.api_key}
                    )
                else:
                    # Fallback to standard client
                    self.client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url
                    )
            else:
                # Standard OpenAI or other endpoints
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
        elif self.provider == "anthropic":
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = Anthropic(**client_kwargs)
        elif self.provider == "gemini":
            self._load_gemini_modules()
            # google.genai Client does not accept client_options; use base_url via env if needed.
            if self.base_url:
                os.environ["GOOGLE_API_BASE_URL"] = self.base_url
            self.client = self._genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown provider for model {self.model_name}")

    def _determine_provider(self, model_name: str) -> str:
        """Infer provider from the model name."""
        name = model_name.lower()
        if name.startswith("claude"):
            return "anthropic"
        if name.startswith("gemini"):
            return "gemini"
        return "openai"

    def _load_gemini_modules(self):
        """Lazy-load google-genai modules to avoid hard dependency for non-Gemini runs."""
        if hasattr(self, "_genai") and self._genai is not None:
            return
        try:
            self._genai = importlib.import_module("google.genai")
            self._genai_types = importlib.import_module("google.genai.types")
            self._client_options_lib = importlib.import_module("google.api_core.client_options")
        except Exception as e:
            raise ImportError(
                "google-genai is required for Gemini models. Install with `pip install google-genai`."
            ) from e

    def _is_reasoning_model(self) -> bool:
        """
        Returns True if the configured model uses the reasoning API contract
        (e.g. gpt-5/o-series) that expects max_completion_tokens instead of max_tokens.
        """
        if not self.model_name:
            return False
        name = self.model_name.lower()
        reasoning_keywords = (
            "o1",
            "o3",
            "o4",
            "gpt-4.1",
            "gpt5",
            "gpt-5",
        )
        return any(keyword in name for keyword in reasoning_keywords)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text

        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If stream=False, returns the generated text
            If stream=True, returns the streaming response object
        """

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens

        if self.provider == "openai":
            return self._openai_generate(messages, temp, tokens, stops, stream, **kwargs)
        if self.provider == "anthropic":
            return self._anthropic_generate(messages, temp, tokens, stops, stream, **kwargs)
        if self.provider == "gemini":
            return self._gemini_generate(messages, temp, tokens, stops, stream)
        raise ValueError(f"Unsupported provider {self.provider}")

    def _openai_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop_tokens: Optional[Union[str, List[str]]],
        stream: bool,
        **kwargs,
    ):
        is_reasoning_model = self._is_reasoning_model()
        if is_reasoning_model:
            temperature = 1.0
            kwargs.pop("temperature", None)

        normalized_messages = self._normalize_messages(messages)
        params = {
            "model": self.model_name,
            "messages": normalized_messages,
            "stream": stream,
            **kwargs
        }

        if is_reasoning_model:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature

        if stop_tokens is not None:
            params["stop"] = stop_tokens

        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.chat.completions.create(**params)
                if stream:
                    return response
                return response.choices[0].message
            except Exception as e:
                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)

    def _anthropic_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop_tokens: Optional[Union[str, List[str]]],
        stream: bool,
        **kwargs,
    ):
        system = None
        converted_messages = []

        def _to_blocks(content):
            if isinstance(content, list):
                content = "\n".join([str(c) for c in content])
            return [{"type": "text", "text": str(content)}]

        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                converted_messages.append(
                    {"role": m["role"], "content": _to_blocks(m["content"])}
                )

        params = {
            "model": self.model_name,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system:
            params["system"] = system
        if stop_tokens is not None:
            params["stop_sequences"] = stop_tokens
        if stream:
            params["stream"] = True
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.messages.create(**params)
                if stream:
                    text = ""
                    for event in response:
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "text", None):
                            text += delta.text
                    return text

                text_parts = [
                    part.text for part in response.content if getattr(part, "text", None)
                ]
                return "".join(text_parts)
            except Exception as e:
                attempt += 1
                logger.warning(f"calling anthropic model failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)

    def _gemini_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop_tokens: Optional[Union[str, List[str]]],
        stream: bool,
    ):
        self._load_gemini_modules()
        prompt_parts = []
        for message in messages:
            prefix = f"{message['role']}: "
            prompt_parts.append(prefix + message["content"])
        prompt = "\n".join(prompt_parts)

        config_obj = self._genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop_tokens if stop_tokens is not None else None,
        )

        def _extract_text(resp):
            try:
                candidates = getattr(resp, "candidates", None) or []
                if not candidates:
                    return getattr(resp, "text", "") or ""
                parts = getattr(candidates[0], "content", None)
                parts = getattr(parts, "parts", []) if parts else []
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                return "".join(texts) if texts else getattr(resp, "text", "") or ""
            except Exception:
                return getattr(resp, "text", "") or ""

        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                    config=config_obj,
                )
                if stream:
                    return _extract_text(response)

                return _extract_text(response)
            except Exception as e:
                attempt += 1
                logger.warning(f"calling gemini model failed, retrying {attempt}/retry, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        delay_time:int = 1,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text

        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If stream=False, returns the generated text
            If stream=True, returns the streaming response object
        """
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        return self.generate(
            messages,
            temperature=temp,
            max_tokens=tokens,
            stop_tokens=stops,
            stream=True,
            **kwargs,
        )
    
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns a streaming response object
        """

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens

        if self.provider != "openai":
            result = self.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=tokens,
                stop_tokens=stops,
                stream=stream,
                **kwargs,
            )
            return result

        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
            **kwargs
        }
        if stops is not None:
            params["stop"] = stops

        response = self.client.completions.create(**params)
        if stream:
            return response
        return response.choices[0].text
    
    def stream_complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion - Auto-detects and uses appropriate API

        Supports both:
        - Completions API (for models like original deepseek-r1)
        - Chat Completions API (for models like deepseek-reasoner)
        - Special handling for o4-mini/o1/o3 models

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            Generated text in format: "<think>reasoning</think>\nfinal_answer"
            (reasoning_content models converted to this format automatically)
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True

        if self.provider != "openai":
            return self.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=tokens,
                stop_tokens=stops,
                stream=True,
                **kwargs,
            )

        # Check if model is a reasoning model (gpt-5/o-series have special requirements)
        is_reasoning_model = self._is_reasoning_model()
        if is_reasoning_model:
            temp = 1.0
            kwargs.pop("temperature", None)

        attempt = 0
        while attempt < self.retry_time:
            try:
                # Try chat completions API first (for deepseek-reasoner models)
                params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": stream,
                    **kwargs
                }

                # Reasoning models use max_completion_tokens instead of max_tokens
                if is_reasoning_model:
                    params["max_completion_tokens"] = tokens
                    # Don't set temperature for reasoning models
                else:
                    params["max_tokens"] = tokens
                    params["temperature"] = temp

                if stops is not None:
                    params["stop"] = stops

                response = self.client.chat.completions.create(**params)

                reasoning_text = ""
                content_text = ""

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        # Handle reasoning_content field (for deepseek-reasoner)
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                            reasoning_text += delta.reasoning_content
                        # Handle regular content field
                        if hasattr(delta, 'content') and delta.content is not None:
                            content_text += delta.content

                # Combine in <think> format for compatibility
                if reasoning_text:
                    full_text = f"<think>\n{reasoning_text}\n</think>\n{content_text}"
                else:
                    # If no reasoning_content, check if content has <think> tags already
                    full_text = content_text

                return full_text

            except Exception as e:
                # Check if error is about chat API not supported
                error_str = str(e).lower()
                if "chat" in error_str or attempt >= self.retry_time - 1:
                    # Fallback to completions API (for older models)
                    try:
                        logger.info("Falling back to completions API")
                        params = {
                            "model": self.model_name,
                            "prompt": prompt,
                            "temperature": temp,
                            "max_tokens": tokens,
                            "stream": stream,
                            **kwargs
                        }

                        if stops is not None:
                            params["stop"] = stops

                        response = self.client.completions.create(**params)

                        full_text = ""
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].text is not None:
                                full_text += chunk.choices[0].text
                        return full_text
                    except Exception as e2:
                        logger.error(f"Both chat and completions API failed: {e2}")
                        raise e2

                attempt += 1
                logger.warning(f"calling llm failed, retrying {attempt}/{self.retry_time}, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                time.sleep(self.delay_time)
    
