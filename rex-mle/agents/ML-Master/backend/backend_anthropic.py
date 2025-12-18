"""Backend for Anthropic Claude models."""

import logging
import time
import json

import anthropic
from funcy import notnone, select_values

from backend.backend_utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

_client: anthropic.Anthropic | None = None


def _setup_anthropic_client(cfg: Config) -> None:
    """Initialize a shared Anthropic client."""
    global _client

    if _client is not None:
        return

    client_kwargs = {"api_key": cfg.agent.feedback.api_key}
    # Anthropic supports custom base URLs for proxies; leave empty to use default.
    if cfg.agent.feedback.base_url:
        client_kwargs["base_url"] = cfg.agent.feedback.base_url

    _client = anthropic.Anthropic(**client_kwargs)


def _convert_messages(
    system_message: str | None, user_message: str | None, convert_system_to_user: bool
) -> tuple[str | None, list[dict[str, str]]]:
    """Convert OpenAI-style messages into Anthropic format."""
    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )
    system = None
    converted: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    return system, converted


def _ensure_anthropic_content_format(messages: list[dict[str, str]]) -> list[dict]:
    """Anthropic SDK expects list-of-blocks content; wrap everything into a single text block."""
    converted: list[dict] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            # Flatten any list by stringifying and joining with newlines
            content = "\n".join([str(c) for c in content])
        content_str = str(content)
        converted.append(
            {
                "role": m["role"],
                "content": [{"type": "text", "text": content_str}],
            }
        )
    return converted


def _build_tools(func_spec: FunctionSpec | None):
    if func_spec is None:
        return None, None
    tools = [
        {
            "name": func_spec.name,
            "description": func_spec.description,
            "input_schema": func_spec.json_schema,
        }
    ]
    tool_choice = {"type": "tool", "name": func_spec.name}
    return tools, tool_choice


def _parse_tool_output(completion, func_spec: FunctionSpec) -> dict:
    """Extract tool call payload from Anthropic response."""
    tool_blocks = [
        block
        for block in completion.content
        if getattr(block, "type", None) == "tool_use"
    ]
    if not tool_blocks:
        raise ValueError(
            f"No tool_use blocks returned for function {func_spec.name}: {completion}"
        )
    target_block = None
    for block in tool_blocks:
        if getattr(block, "name", None) == func_spec.name:
            target_block = block
            break
    if target_block is None:
        target_block = tool_blocks[0]
    return target_block.input  # type: ignore[attr-defined]


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Run a query against Anthropic models."""
    if cfg is None:
        raise ValueError("Config is required for Anthropic backend.")
    _setup_anthropic_client(cfg)
    assert _client is not None

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    temperature = filtered_kwargs.pop("temperature", None)
    max_tokens = filtered_kwargs.pop("max_tokens", None)
    stop_sequences = filtered_kwargs.pop("stop", None)

    system, messages = _convert_messages(
        system_message, user_message, convert_system_to_user
    )
    # If there is no user message, fallback to putting the system content as user content
    if not messages and system:
        messages = [{"role": "user", "content": system}]
        system = None
    messages = _ensure_anthropic_content_format(messages)
    logger.info(f"[anthropic] sending messages={messages} system={system}")
    tools, tool_choice = _build_tools(func_spec)

    t0 = time.time()
    logger.info("Querying Anthropic model for feedback stage")
    completion = backoff_create(
        _client.messages.create,
        (anthropic.RateLimitError, anthropic.APIStatusError),
        model=filtered_kwargs["model"],
        messages=messages,
        system=system,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
    )
    req_time = time.time() - t0

    if func_spec is None:
        text_parts = [
            block.text for block in completion.content if getattr(block, "text", None)
        ]
        output: OutputType = "".join(text_parts)
    else:
        output = _parse_tool_output(completion, func_spec)

    in_tokens = getattr(completion, "usage", None)
    prompt_tokens = getattr(in_tokens, "input_tokens", None) or 0
    completion_tokens = getattr(in_tokens, "output_tokens", None) or 0

    info = {
        "id": getattr(completion, "id", None),
        "model": getattr(completion, "model", None),
    }

    return output, req_time, prompt_tokens, completion_tokens, info
