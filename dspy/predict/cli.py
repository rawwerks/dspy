"""Run one DSPy signature through an arbitrary stdin/stdout command."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import dspy
from dspy.clients.cli_command import CLICommand
from dspy.clients.cli_process import DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT_SECONDS
from dspy.clients.cli_stdio import CLIError, _StdioCLI
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.annotation import experimental


@experimental
class CLI(Module):
    """Use a command as the execution backend for one DSPy predictor.

    The contained :class:`dspy.Predict` is the sole optimizer target. DSPy's
    adapter renders its signature and inputs, the resulting text is written to
    the command's stdin, and the adapter parses stdout into typed signature
    outputs. The command remains an opaque argv sequence; the module never uses
    or modifies the globally configured LM, and the command is reconstructed
    from ``__init__`` rather than serialized into saved program state.

    Unless an adapter is configured globally, calls use a ``ChatAdapter`` with
    the JSON-fallback retry disabled, so the command runs at most once per
    generation. A globally configured adapter is honored as-is; if it retries
    on parse failure, the command runs again.

    Args:
        signature: The signature of the predictor, as a class or shorthand string.
        command: Argv sequence (or :class:`dspy.CLICommand`) to execute per
            generation. Never a shell string.
        cwd: Working directory for the command. Defaults to the current one.
        env: Environment variables merged over the inherited environment.
        timeout: Seconds before the command's process group is killed.
            Defaults to 600; pass ``None`` for no limit. Agent CLIs can block
            indefinitely waiting for interaction the transport cannot provide,
            so an unbounded call is opt-in.
        encoding: Text encoding for stdin and stdout. Defaults to UTF-8.
        max_output_bytes: Independent cap on captured stdout and stderr.

    Example:
        ``dspy.CLI("question -> answer", dspy.resolve_agent_cli("claude"))``

    Warning:
        The command runs on the host with the current process's authority. This
        module is not a sandbox: the child process inherits the full parent
        environment (including any secrets in it), and agent CLIs additionally
        apply their own host-level configuration, such as user permission
        allowlists. Grant authority explicitly with the agent's own flags, and
        treat untrusted signature inputs as prompt injection against whatever
        tools the agent may use.
    """

    def __init__(
        self,
        signature: type[dspy.Signature] | str,
        command: Sequence[str] | CLICommand,
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
        encoding: str = "utf-8",
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        super().__init__()
        if isinstance(command, CLICommand):
            if command.prompt_transport != "stdin" or command.output_format != "text":
                raise ValueError("CLI currently supports only text over stdin/stdout")
            command = command.argv
        self.predict = Predict(signature)
        self._transport = _StdioCLI(
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            encoding=encoding,
            max_output_bytes=max_output_bytes,
        )
        self._adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)

    def forward(self, **inputs: Any) -> Prediction:
        with dspy.context(adapter=dspy.settings.adapter or self._adapter):
            return self.predict(**inputs, lm=self._transport)

    async def aforward(self, **inputs: Any) -> Prediction:
        with dspy.context(adapter=dspy.settings.adapter or self._adapter):
            return await self.predict.acall(**inputs, lm=self._transport)


__all__ = ["CLI", "CLIError"]
