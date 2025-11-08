from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dspy.clients.cli_lm import CLILM, CLILMError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cli_echo.py"


def _make_lm(env: dict[str, str] | None = None) -> CLILM:
    return CLILM([sys.executable, str(SCRIPT)], env=env)


def _messages(prompt: str):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": prompt},
    ]


def test_cli_lm_basic_round_trip():
    lm = _make_lm()
    outputs = lm(prompt=None, messages=_messages("hello"))
    assert outputs[0] == "hello"


def test_cli_lm_handles_n_parameter():
    lm = _make_lm()
    outputs = lm(prompt=None, messages=_messages("multi"), n=2)
    assert outputs == ["multi", "multi"]


@pytest.mark.asyncio
async def test_cli_lm_async_round_trip():
    lm = _make_lm()
    outputs = await lm.acall(prompt=None, messages=_messages("async"))
    assert outputs[0] == "async"


def test_cli_lm_parses_codex_jsonl():
    lm = _make_lm(env={"CLI_MODE": "json"})
    outputs = lm(prompt=None, messages=_messages("json output"))
    assert outputs[0] == "json output"


def test_cli_lm_raises_on_failure():
    lm = _make_lm(env={"CLI_MODE": "fail"})
    with pytest.raises(CLILMError):
        lm(prompt=None, messages=_messages("boom"))


def test_cli_lm_exposes_raw_stdout_if_not_json():
    lm = _make_lm(env={"CLI_MODE": "invalid"})
    outputs = lm(prompt=None, messages=_messages("fallback"))
    assert outputs[0] == "{not json"
