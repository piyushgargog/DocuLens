"""Tests marked with `requires_api_key` make a real call to the configured
LLM API and are skipped automatically when LLM_API_KEY isn't set (e.g. in
CI, which has no secret configured)."""

import os

import pytest

from llm_client import LLMConfigError, ask

requires_api_key = pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set; skipping tests that call a real LLM API",
)


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        ask("does this raise?", [])


@requires_api_key
def test_answers_from_the_provided_passages():
    passages = [
        {"page": 3, "text": "Jupiter is the largest planet in the Solar System.", "score": 0.9}
    ]
    answer = ask("Which planet is the largest?", passages)
    assert "jupiter" in answer.lower()


@requires_api_key
def test_refuses_when_passages_dont_support_an_answer():
    passages = [
        {"page": 3, "text": "Jupiter is the largest planet in the Solar System.", "score": 0.9}
    ]
    answer = ask("What is the capital of France?", passages)
    assert "could not find" in answer.lower()


def test_single_turn_prompt_is_unchanged_without_history():
    from llm_client import SYSTEM_PROMPT, build_messages

    messages = build_messages("q?", [{"page": 1, "text": "t"}])
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert len(messages) == 2
    assert "[Page 1] t" in messages[1]["content"]


def test_history_turns_and_document_labels_are_included():
    from llm_client import HISTORY_RULE, build_messages

    history = [{"question": "first?", "answer": "first answer"}]
    messages = build_messages("second?", [{"page": 2, "text": "t", "doc": "a.pdf"}], history)
    assert messages[0]["content"].endswith(HISTORY_RULE)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "first answer"
    assert "[a.pdf, Page 2] t" in messages[3]["content"]
