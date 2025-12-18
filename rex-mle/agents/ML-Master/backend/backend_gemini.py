"""Backend for Google Gemini models (google-genai)."""

import importlib
import json
import logging
import os
import time
from typing import Any, Tuple

from funcy import notnone, select_values

from backend.backend_utils import FunctionSpec, OutputType
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

_client = None
_genai = None
_genai_types = None


def _load_genai_modules():
    """Lazy-load google-genai modules to avoid hard dependency for non-Gemini runs."""
    global _genai, _genai_types
    if _genai and _genai_types:
        return
    try:
        _genai = importlib.import_module("google.genai")
        _genai_types = importlib.import_module("google.genai.types")
    except Exception as e:
        raise ImportError(
            "google-genai is required for Gemini models. Install with `pip install google-genai`."
        ) from e


def _configure_client(cfg: Config) -> None:
    """Configure the global Gemini client."""
    global _client
    if _client is not None:
        return
    _load_genai_modules()
    if cfg.agent.feedback.base_url:
        os.environ["GOOGLE_API_BASE_URL"] = cfg.agent.feedback.base_url
    _client = _genai.Client(api_key=cfg.agent.feedback.api_key)


def _build_generation_config(temperature: float | None, max_tokens: int | None, stop: Any):
    _load_genai_modules()
    return _genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        stop_sequences=stop if stop is not None else None,
    )


def _build_tooling(func_spec: FunctionSpec | None) -> Tuple[list | None, Any]:
    _load_genai_modules()
    if func_spec is None:
        return None, None
    tool = _genai_types.Tool(
        function_declarations=[
            _genai_types.FunctionDeclaration(
                name=func_spec.name,
                description=func_spec.description,
                parameters=func_spec.json_schema,
            )
        ]
    )
    tool_config = _genai_types.ToolConfig(
        function_calling_config=_genai_types.FunctionCallingConfig(
            mode="ANY", allowed_function_names=[func_spec.name]
        )
    )
    return [tool], tool_config


def _combine_prompt(system_message: str | None, user_message: str | None) -> str:
    def _as_text(p):
        if p is None:
            return None
        if isinstance(p, str):
            return p
        return json.dumps(p, ensure_ascii=False)

    parts = [t for t in [_as_text(system_message), _as_text(user_message)] if t]
    return "\n\n".join(parts)


def _parse_function_call(response, func_spec: FunctionSpec) -> dict:
    candidate = response.candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", []) if content else []
    for part in parts:
        func_call = getattr(part, "function_call", None)
        if not func_call:
            continue
        if func_spec.name and func_call.name != func_spec.name:
            continue
        args = getattr(func_call, "args", None)
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args)
        except Exception:
            return {"raw_arguments": args}
    raise ValueError(
        f"No function call returned for {func_spec.name}: {response.candidates[0]}"
    )

def _extract_text(response) -> str:
    """Safely extract text parts to avoid SDK warnings about non-text parts."""
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return getattr(response, "text", "") or ""
        parts = getattr(candidates[0], "content", None)
        parts = getattr(parts, "parts", []) if parts else []
        texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
        return "".join(texts) if texts else getattr(response, "text", "") or ""
    except Exception:
        return getattr(response, "text", "") or ""


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,  # Unused, included for signature parity
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Run a query against Gemini models."""
    if cfg is None:
        raise ValueError("Config is required for Gemini backend.")
    _configure_client(cfg)
    assert _client is not None

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    temperature = filtered_kwargs.pop("temperature", None)
    max_tokens = filtered_kwargs.pop("max_tokens", None)
    stop_sequences = filtered_kwargs.pop("stop", None)

    generation_config = _build_generation_config(temperature, max_tokens, stop_sequences)
    tools, tool_config = _build_tooling(func_spec)
    prompt = _combine_prompt(system_message, user_message)

    t0 = time.time()
    logger.info("Querying Gemini model for feedback stage")
    # Some SDK versions do not support tool arguments; only include when present and supported.
    generate_kwargs = {
        "model": filtered_kwargs["model"],
        "contents": [prompt],
        "config": generation_config,
    }
    if tools and tool_config:
        generate_kwargs["tools"] = tools
        generate_kwargs["tool_config"] = tool_config
    try:
        response = _client.models.generate_content(**generate_kwargs)
    except TypeError as te:
        # Retry without tooling for SDKs that don't yet support it
        if "tools" in generate_kwargs:
            logger.debug(
                "Gemini SDK does not support tool arguments; retrying without tools. Error: %s",
                te,
            )
            generate_kwargs.pop("tools", None)
            generate_kwargs.pop("tool_config", None)
            response = _client.models.generate_content(**generate_kwargs)
        else:
            raise
    req_time = time.time() - t0

    if func_spec is None:
        output: OutputType = _extract_text(response)
    else:
        # Tool calling may be unsupported in current SDK; fallback to plain text if parsing fails.
        try:
            output = _parse_function_call(response, func_spec)
        except Exception:
            output = _extract_text(response)

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    info = {
        "model": getattr(response, "model", None),
        "finish_reasons": getattr(response.candidates[0], "finish_reason", None)
        if response.candidates
        else None,
    }

    return output, req_time, prompt_tokens, completion_tokens, info
