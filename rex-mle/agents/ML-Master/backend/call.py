import logging

from utils.llm_caller import LLM
from backend.backend_utils import PromptType
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")


def _prompt_to_messages(prompt: PromptType | None) -> list[dict]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, dict):
        return [prompt]
    if isinstance(prompt, list):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")

def r1_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg:Config=None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    logger.info(f"prompt: {prompt}", extra={"verbose": True})
    model_kwargs = model_kwargs | {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = llm.stream_complete(
        prompt,
        **model_kwargs
    )

    # Extract content after </think> tag
    # stream_complete now returns format: "<think>reasoning</think>\nfinal_answer"
    if "</think>" in response:
        res = response[response.find("</think>")+8:].strip()
    else:
        # If no think tags, return the full response
        res = response

    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"response without think:\n{res}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res


def gpt_query(
    prompt: PromptType | None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cfg: Config | None = None,
    **model_kwargs,
):
    llm = LLM(
        base_url=cfg.agent.code.base_url,
        api_key=cfg.agent.code.api_key,
        model_name=cfg.agent.code.model,
    )
    logger.info(f"using {llm.model_name} to generate code.")
    logger.info("---Querying model---", extra={"verbose": True})
    logger.info(f"prompt: {prompt}", extra={"verbose": True})

    model_kwargs = model_kwargs | {"temperature": temperature, "max_tokens": max_tokens}
    if isinstance(prompt, str):
        # For GPT-5/o-series on Azure/OpenAI, stream_complete handles max_completion_tokens
        # and returns a fully aggregated string.
        response = llm.stream_complete(prompt, **model_kwargs)
    else:
        # Treat non-string prompts as chat messages and aggregate the stream to a string.
        stream = llm.stream_generate(_prompt_to_messages(prompt), **model_kwargs)
        reasoning_text = ""
        content_text = ""
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if not delta:
                continue
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_text += delta.reasoning_content
            if hasattr(delta, "content") and delta.content is not None:
                content_text += delta.content
        if reasoning_text:
            response = f"<think>\n{reasoning_text}\n</think>\n{content_text}"
        else:
            response = content_text

    # Normalize output: if a <think> block is present, strip it for downstream parsing.
    if isinstance(response, str) and "</think>" in response:
        res = response[response.find("</think>") + 8 :].strip()
    else:
        res = response

    logger.info(f"response:\n{response}", extra={"verbose": True})
    logger.info(f"---Query complete---", extra={"verbose": True})
    return res
