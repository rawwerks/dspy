from __future__ import annotations

import sys
from pathlib import Path

import pytest

import dspy
from dspy.adapters.cli_adapter import CLIAdapter, CLIAdapterError
from dspy.utils.dummies import DummyLM


FIXTURE = Path(__file__).resolve().parents[2] / "agent.py"


def _make_adapter(mode: str | None = None, **kwargs) -> CLIAdapter:
    command = [sys.executable, str(FIXTURE)]
    if mode:
        command.append(mode)
    return CLIAdapter(command, **kwargs)


def _dummy_lm() -> DummyLM:
    return DummyLM([{"answer": "unused"}])


def test_cli_adapter_sync_round_trip():
    signature = dspy.make_signature("question->answer")
    adapter = _make_adapter("echo")

    result = adapter(_dummy_lm(), {}, signature, [], {"question": "What is 2 + 2?"})

    assert result == [{"answer": "Echo 1: What is 2 + 2?"}]


def test_cli_adapter_respects_n_parameter():
    signature = dspy.make_signature("question->answer")
    adapter = _make_adapter("multi")

    result = adapter(_dummy_lm(), {"n": 2}, signature, [], {"question": "color?"})

    assert [item["answer"] for item in result] == ["Echo 1: color?", "Echo 2: color?"]


@pytest.mark.asyncio
async def test_cli_adapter_async_round_trip():
    signature = dspy.make_signature("question->answer")
    adapter = _make_adapter("echo")

    results = await adapter.acall(_dummy_lm(), {}, signature, [], {"question": "async?"})

    assert results == [{"answer": "Echo 1: async?"}]


def test_cli_adapter_raises_when_command_fails():
    signature = dspy.make_signature("question->answer")
    adapter = _make_adapter("fail")

    with pytest.raises(CLIAdapterError) as exc:
        adapter(_dummy_lm(), {}, signature, [], {"question": "boom"})

    assert "status 2" in str(exc.value)
    assert "intentional failure" in str(exc.value)


def test_cli_adapter_validates_json_shape():
    signature = dspy.make_signature("question->answer")
    adapter = _make_adapter("invalid_json")

    with pytest.raises(CLIAdapterError) as exc:
        adapter(_dummy_lm(), {}, signature, [], {"question": "bad"})

    assert "Could not decode" in str(exc.value)


def test_cli_adapter_supports_custom_env():
    signature = dspy.make_signature("question->answer")
    adapter = _make_adapter("echo", env={"CLI_ADAPTER_ECHO_ENV": "visible"})

    result = adapter(_dummy_lm(), {}, signature, [], {"question": "env"})

    assert result == [{"answer": "Echo 1: env [visible]"}]
