import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from backend import determine_provider


def test_determine_provider_includes_gemini_and_claude():
    assert determine_provider("gemini-3") == "gemini"
    assert determine_provider("gemini-3-pro-preview") == "gemini"
    assert determine_provider("claude-sonnet-4.5") == "anthropic"
    assert determine_provider("claude-sonnet-4-5-20250929") == "anthropic"
    assert determine_provider("gpt-4o") == "openai"
