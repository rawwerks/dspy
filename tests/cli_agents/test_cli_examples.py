from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from dspy.clients.cli_lm import CLILM
from tests.cli_agents.constants import CLI_LIST

pytestmark = pytest.mark.cli_agent
RUN_CLI_AGENT_TESTS = bool(os.environ.get("DSPY_RUN_CLI_AGENT_TESTS"))


@pytest.fixture(autouse=True)
def require_real_cli_agents():
    if not RUN_CLI_AGENT_TESTS:
        pytest.skip("Set DSPY_RUN_CLI_AGENT_TESTS=1 to run CLI agent tests.")

CLI_ENV_OVERRIDES = {
    "claude -p": "DSPY_CLI_COMMAND_CLAUDE",
    "codex exec": "DSPY_CLI_COMMAND_CODEX",
}

CLI_DEFAULTS = {
    "claude -p": ["claude", "-p"],
    "codex exec": ["codex", "exec"],
}


@dataclass(frozen=True)
class CLIExample:
    name: str
    base_command: str
    extra_args: tuple[str, ...]
    question: str
    uses_messages: bool = True


CLI_EXAMPLES = [
    CLIExample(
        name="claude_text_stdin",
        base_command="claude -p",
        extra_args=(
            "--dangerously-skip-permissions",
        ),
        question="What is 2 + 2?",
    ),
    CLIExample(
        name="codex_text_stdin",
        base_command="codex exec",
        extra_args=(),
        question="List the project files",
    ),
]


def _ensure_command_available(executable: str, env_var: str | None, base_command: str) -> None:
    if shutil.which(executable) is not None:
        return
    if Path(executable).exists():
        return
    hint = f"Install '{base_command}' or set {env_var} to point to the CLI binary." if env_var else ""
    pytest.skip(f"{base_command} executable '{executable}' not found. {hint}".strip())


def _resolve_base_command(base_command: str) -> list[str]:
    env_var = CLI_ENV_OVERRIDES.get(base_command)
    override = os.getenv(env_var) if env_var else None
    if override:
        tokens = shlex.split(override)
    else:
        tokens = list(CLI_DEFAULTS[base_command])
    _ensure_command_available(tokens[0], env_var, base_command)
    return tokens


def _command_for(example: CLIExample) -> list[str]:
    return _resolve_base_command(example.base_command) + list(example.extra_args)


def _build_messages(question: str):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": question},
    ]


@pytest.mark.parametrize("example", CLI_EXAMPLES, ids=lambda example: example.name)
def test_cli_tutorial_examples_require_real_commands(example: CLIExample):
    command = _command_for(example)
    lm = CLILM(command)
    outputs = (
        lm(prompt=None, messages=_build_messages(example.question))
        if example.uses_messages
        else lm(prompt=example.question, messages=None)
    )
    assert isinstance(outputs[0], str)
    assert outputs[0].strip()


def test_cli_list_matches_examples():
    bases = sorted({example.base_command for example in CLI_EXAMPLES})
    assert bases == CLI_LIST
