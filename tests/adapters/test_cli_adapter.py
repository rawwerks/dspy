from __future__ import annotations

import sys
from pathlib import Path

import pytest

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.cli_lm import CLILM, CLILMError


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "stub_cli_agent.py"


def _make_lm(env: dict[str, str] | None = None) -> CLILM:
    return CLILM([sys.executable, str(FIXTURE)], env=env)


def test_cli_adapter_sync_round_trip():
    signature = dspy.make_signature("question->answer")
    adapter = ChatAdapter()
    lm = _make_lm()

    result = adapter(lm, {}, signature, [], {"question": "What is 2 + 2?"})

    assert result == [{"answer": "Echo 1: What is 2 + 2?"}]


def test_cli_adapter_respects_n_parameter():
    signature = dspy.make_signature("question->answer")
    adapter = ChatAdapter()
    lm = _make_lm()

    result = adapter(lm, {"n": 2}, signature, [], {"question": "color?"})

    assert [item["answer"] for item in result] == ["Echo 1: color?", "Echo 2: color?"]


@pytest.mark.asyncio
async def test_cli_adapter_async_round_trip():
    signature = dspy.make_signature("question->answer")
    adapter = ChatAdapter()
    lm = _make_lm()

    results = await adapter.acall(lm, {}, signature, [], {"question": "async?"})

    assert results == [{"answer": "Echo 1: async?"}]


def test_cli_adapter_raises_when_command_fails():
    signature = dspy.make_signature("question->answer")
    adapter = ChatAdapter()
    lm = _make_lm(env={"CLI_MODE": "fail"})

    with pytest.raises(CLILMError) as exc:
        adapter(lm, {}, signature, [], {"question": "boom"})

    assert "status 2" in str(exc.value)
    assert "intentional failure" in str(exc.value)


def test_cli_adapter_validates_json_shape():
    signature = dspy.make_signature("question->answer")
    adapter = ChatAdapter()
    lm = _make_lm(env={"CLI_MODE": "json"})

    result = adapter(lm, {}, signature, [], {"question": "structured?"})

    assert result == [{"answer": "Echo 1: structured?"}]


def test_cli_adapter_supports_custom_env():
    signature = dspy.make_signature("question->answer")
    adapter = ChatAdapter()
    lm = _make_lm(env={"CLI_ADAPTER_ECHO_ENV": "visible"})

    result = adapter(lm, {}, signature, [], {"question": "env"})

    assert result == [{"answer": "Echo 1: env [visible]"}]
