"""Tests for src/agent.py — mocked Claude; no live API calls."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agent import (
    call_claude,
    call_llm,
    get_claude_client,
    parse_problem,
    run_agent,
    run_benchmark,
)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_request: dict | None = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return _FakeResponse(self.response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text: str = "Shift budget toward high-ROAS channels."):
        self.messages = _FakeMessages(response_text)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Prevent tests from picking up the developer's local env file."""
    monkeypatch.setattr("src.agent._load_env_files", lambda: [])
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY", "API_Key"):
        monkeypatch.delenv(name, raising=False)


def test_get_claude_client_returns_none_without_api_key():
    assert get_claude_client() is None


def test_get_claude_client_returns_client_with_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_client = object()
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)

    client = get_claude_client()
    assert client is fake_client
    fake_anthropic.Anthropic.assert_called_once_with(api_key="test-key")


def test_call_claude_returns_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.agent.get_claude_client", lambda: _FakeAnthropicClient("Claude reply"))

    text = call_claude(
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="system",
    )
    assert text == "Claude reply"


def test_call_llm_uses_ollama_when_configured(monkeypatch):
    calls = {}

    class _FakeOllamaResponse:
        def __init__(self):
            self.status_code = 200
            self._json = {
                "message": {"content": "Local model reply"},
            }

        def json(self):
            return self._json

    def _fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return _FakeOllamaResponse()

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr("src.agent.httpx.post", _fake_post)

    text = call_llm(
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="system",
    )
    assert text == "Local model reply"
    assert calls["url"] == "http://localhost:11434/api/chat"
    assert calls["json"]["model"] == "llama3.1:8b"


def test_parse_problem_detects_upload_intent():
    parsed = parse_problem("How do I upload my zip dataset?", {"phase": "upload_request"})
    assert parsed["primary_intent"] == "upload"
    assert "upload" in parsed["intents"]


def test_parse_problem_marks_optimize_blocked_without_confirmation():
    parsed = parse_problem(
        "Run budget optimization across channels",
        {"backward_analysis_confirmed": False},
    )
    assert parsed["primary_intent"] == "optimize"
    assert parsed["optimization_blocked"] is True
    assert parsed["blocked_reason"] == "backward_analysis_not_confirmed"


def test_parse_problem_allows_optimize_after_confirmation():
    parsed = parse_problem(
        "Run budget optimization across channels",
        {"backward_analysis_confirmed": True},
    )
    assert parsed["blocked_reason"] is None


def test_run_agent_returns_fallback_without_api_key():
    response = run_agent(
        "What is marketing mix modeling?",
        "system prompt",
        [],
        context={"backward_analysis_confirmed": True},
    )
    assert "ANTHROPIC_API_KEY" in response


def test_run_agent_blocks_optimize_before_backward_analysis():
    response = run_agent(
        "Please optimize my channel allocation",
        "system prompt",
        [],
        context={"backward_analysis_confirmed": False},
    )
    assert "backward analysis" in response.lower()


def test_run_agent_uses_claude_when_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = _FakeAnthropicClient("Claude reply")
    monkeypatch.setattr("src.agent.get_claude_client", lambda: fake_client)

    response = run_agent(
        "Explain my google_paid_search spend",
        "system prompt",
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        context={"phase": "analysis"},
    )
    assert response == "Claude reply"
    assert fake_client.messages.last_request is not None
    assert fake_client.messages.last_request["system"] == "system prompt"
    assert fake_client.messages.last_request["messages"][-1]["content"] == (
        "Explain my google_paid_search spend"
    )


def test_run_agent_deduplicates_trailing_user_turn(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_client = _FakeAnthropicClient("ok")
    monkeypatch.setattr("src.agent.get_claude_client", lambda: fake_client)

    run_agent(
        "What channels matter most?",
        "system prompt",
        [{"role": "user", "content": "What channels matter most?"}],
    )
    assert fake_client.messages.last_request["messages"] == [
        {"role": "user", "content": "What channels matter most?"}
    ]


def test_run_agent_handles_empty_message():
    assert "type a message" in run_agent("", "system", []).lower()


def test_run_benchmark_collects_results(monkeypatch):
    monkeypatch.setattr(
        "src.agent.run_agent",
        lambda user_message, system_prompt, conversation_history, context=None: f"echo: {user_message}",
    )

    report = run_benchmark(
        ["upload my dataset", "explain backward analysis"],
        system_prompt="benchmark prompt",
    )

    assert report["n_messages"] == 2
    assert report["n_success"] == 2
    assert report["avg_latency_s"] >= 0
    assert report["results"][0]["response"].startswith("echo:")


def test_run_benchmark_records_failures(monkeypatch):
    def _boom(user_message, system_prompt, conversation_history, context=None):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("src.agent.run_agent", _boom)

    report = run_benchmark(["trigger failure"])
    assert report["n_success"] == 0
    assert report["n_failed"] == 1
    assert "model unavailable" in report["results"][0]["error"]
