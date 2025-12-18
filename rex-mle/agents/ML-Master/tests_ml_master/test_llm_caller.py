import sys
from pathlib import Path
from types import SimpleNamespace

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from utils.llm_caller import LLM


class DummyResponse:
    def __init__(self):
        self.choices = [SimpleNamespace(message="ok")]


class DummyCompletions:
    def __init__(self):
        self.last_params = None

    def create(self, **params):
        self.last_params = params
        return DummyResponse()


def _build_dummy_client():
    completions = DummyCompletions()
    chat = SimpleNamespace(completions=completions)
    client = SimpleNamespace(chat=chat)
    return client, completions


def test_gpt5_uses_max_completion_tokens():
    llm = LLM(
        base_url="http://localhost",
        api_key="dummy",
        model_name="gpt-5",
        max_tokens=128,
    )
    client, completions = _build_dummy_client()
    llm.client = client

    llm.generate([{"role": "user", "content": "ping"}])

    assert completions.last_params["max_completion_tokens"] == 128
    assert "max_tokens" not in completions.last_params
    assert "temperature" not in completions.last_params


def test_non_reasoning_models_use_standard_max_tokens():
    llm = LLM(
        base_url="http://localhost",
        api_key="dummy",
        model_name="gpt-4o",
        max_tokens=256,
        temperature=0.5,
    )
    client, completions = _build_dummy_client()
    llm.client = client

    llm.generate([{"role": "user", "content": "ping"}])

    assert completions.last_params["max_tokens"] == 256
    assert completions.last_params["temperature"] == 0.5
    assert "max_completion_tokens" not in completions.last_params


if __name__ == "__main__":
    test_gpt5_uses_max_completion_tokens()
    test_non_reasoning_models_use_standard_max_tokens()
    print("All LLM caller tests passed.")
