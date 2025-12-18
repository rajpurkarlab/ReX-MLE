"""Backend for OpenAI API and Azure OpenAI API."""

import json
import logging
import time
import os

from backend.backend_utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai
from openai import AzureOpenAI
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

_client: openai.OpenAI = None  # type: ignore


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _is_azure_endpoint(base_url: str) -> bool:
    """Check if the base_url is an Azure endpoint."""
    if not base_url:
        return False
    azure_indicators = ['azure', '.hms.edu', 'azurewebsites', 'openai.azure.com']
    return any(indicator in base_url.lower() for indicator in azure_indicators)


@once
def _setup_openai_client(cfg:Config):
    global _client

    base_url = cfg.agent.feedback.base_url
    api_key = cfg.agent.feedback.api_key

    # Check if this is an Azure OpenAI endpoint
    if _is_azure_endpoint(base_url):
        logger.info(f"Using Azure OpenAI client for feedback model")
        # Azure OpenAI requires api_version
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        _client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=base_url,
            max_retries=0,
        )
    else:
        logger.info(f"Using standard OpenAI client for feedback model")
        _client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
        )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    cfg:Config=None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    message_print = messages[0]["content"]
    print(f"\033[31m{message_print}\033[0m")
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
        print(f"\033[32m{output}\033[0m")
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
            print(f"\033[32m{output}\033[0m")
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
