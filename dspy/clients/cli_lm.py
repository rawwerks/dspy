"""Language model client that proxies DSPy calls through arbitrary CLI commands."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import time
import uuid
from typing import Sequence

from litellm.utils import Choices, Message, ModelResponse

from dspy.clients.base_lm import BaseLM


class CLILMError(RuntimeError):
    """Raised when the CLI process fails or produces an unexpected response."""

    def __init__(self, message: str, *, stdout: str | None = None, stderr: str | None = None):
        details = [message]
        if stdout is not None:
            details.append(f"stdout:\n{stdout.strip()}" if stdout.strip() else "stdout: <empty>")
        if stderr is not None:
            details.append(f"stderr:\n{stderr.strip()}" if stderr.strip() else "stderr: <empty>")
        super().__init__("\n".join(details))
        self.stdout = stdout
        self.stderr = stderr


class CLILM(BaseLM):
    """BaseLM implementation that communicates with CLI programs via stdin/stdout.

    Example:

        ```python
        import dspy

        lm = dspy.CLILM("python my_cli_model.py")
        dspy.configure(lm=lm)
        print(dspy.Predict("question -> answer")(question="What is 2 + 2?"))
        ```
    """

    def __init__(
        self,
        cli_command: Sequence[str] | str,
        model: str = "cli",
        model_type: str = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = False,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )

        if isinstance(cli_command, str):
            cli_command = shlex.split(cli_command)
        if not cli_command:
            raise ValueError("cli_command cannot be empty")

        self.cli_command = list(cli_command)
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.encoding = encoding

    def forward(self, prompt=None, messages=None, **kwargs):
        prompt_text = self._messages_to_prompt(messages, prompt)
        n = kwargs.get("n") or self.kwargs.get("n") or 1
        outputs = [self._invoke_cli(prompt_text, generation_index=i, total=n) for i in range(n)]
        return self._build_model_response(outputs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        prompt_text = self._messages_to_prompt(messages, prompt)
        n = kwargs.get("n") or self.kwargs.get("n") or 1
        outputs = []
        for i in range(n):
            outputs.append(await self._invoke_cli_async(prompt_text, generation_index=i, total=n))
        return self._build_model_response(outputs)

    # ------------------------------------------------------------------
    # Prompt / response helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _messages_to_prompt(messages, prompt) -> str:
        conversation = messages or ([{"role": "user", "content": prompt}] if prompt else [])
        if not conversation:
            raise CLILMError("No prompt or messages provided to CLILM")

        parts = []
        for message in conversation:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def _build_model_response(self, outputs: list[str]) -> ModelResponse:
        choices = []
        for index, text in enumerate(outputs):
            message = Message(role="assistant", content=text)
            choices.append(Choices(index=index, finish_reason="stop", message=message))

        return ModelResponse(
            id=str(uuid.uuid4()),
            model=self.model,
            created=int(time.time()),
            choices=choices,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            system_fingerprint="cli-lm",
        )

    # ------------------------------------------------------------------
    # CLI invocation
    # ------------------------------------------------------------------
    def _invoke_cli(self, prompt_text: str, *, generation_index: int, total: int) -> str:
        env = self._cli_env(generation_index, total)
        try:
            completed = subprocess.run(
                self.cli_command,
                input=prompt_text,
                capture_output=True,
                text=True,
                encoding=self.encoding,
                cwd=self.cwd,
                env=env,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CLILMError(f"CLI command not found: {self._command_display()}") from exc
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing dependent
            raise CLILMError(
                f"CLI command '{self._command_display()}' timed out after {self.timeout} seconds",
                stdout=exc.stdout,
                stderr=exc.stderr,
            ) from exc

        if completed.returncode != 0:
            raise CLILMError(
                f"CLI command '{self._command_display()}' exited with status {completed.returncode}",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        return self._normalize_output(completed.stdout)

    async def _invoke_cli_async(self, prompt_text: str, *, generation_index: int, total: int) -> str:
        env = self._cli_env(generation_index, total)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.cli_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CLILMError(f"CLI command not found: {self._command_display()}") from exc

        input_bytes = prompt_text.encode(self.encoding)
        communicate_coro = process.communicate(input_bytes)
        try:
            if self.timeout is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(communicate_coro, timeout=self.timeout)
            else:
                stdout_bytes, stderr_bytes = await communicate_coro
        except asyncio.TimeoutError as exc:  # pragma: no cover
            process.kill()
            await process.communicate()
            raise CLILMError(
                f"CLI command '{self._command_display()}' timed out after {self.timeout} seconds"
            ) from exc

        stdout = stdout_bytes.decode(self.encoding, errors="replace")
        stderr = stderr_bytes.decode(self.encoding, errors="replace")
        if process.returncode != 0:
            raise CLILMError(
                f"CLI command '{self._command_display()}' exited with status {process.returncode}",
                stdout=stdout,
                stderr=stderr,
            )
        return self._normalize_output(stdout)

    def _normalize_output(self, raw_stdout: str) -> str:
        raw_stdout = raw_stdout.strip()
        json_message = self._extract_agent_message_from_jsonl(raw_stdout)
        if json_message:
            return json_message
        return raw_stdout

    @staticmethod
    def _extract_agent_message_from_jsonl(stdout: str) -> str | None:
        messages: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    messages.append(text.strip())
        return messages[-1] if messages else None

    def _cli_env(self, generation_index: int, total: int) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env)
        env["CLI_GENERATION_INDEX"] = str(generation_index)
        env["CLI_TOTAL_GENERATIONS"] = str(total)
        return env

    def _command_display(self) -> str:
        try:
            return shlex.join(self.cli_command)
        except AttributeError:  # pragma: no cover - safety
            return " ".join(self.cli_command)

    def dump_state(self) -> dict[str, object]:
        state_keys = [
            "model",
            "model_type",
            "cache",
            "cli_command",
            "env",
            "cwd",
            "timeout",
            "encoding",
        ]
        state = {key: getattr(self, key) for key in state_keys}
        state["cli_command"] = list(self.cli_command)
        state["env"] = dict(self.env)
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != "api_key"}
        return state | filtered_kwargs
