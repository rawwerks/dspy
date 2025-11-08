from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from typing import Any, Sequence

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.utils import get_annotation_name
from dspy.signatures.signature import Signature


class CLIAdapterError(RuntimeError):
    """Raised when the CLI process fails or returns malformed output."""

    def __init__(self, message: str, *, stdout: str | None = None, stderr: str | None = None):
        details = [message]
        if stdout:
            details.append(f"stdout:\n{stdout.strip()}" if stdout.strip() else "stdout: <empty>")
        if stderr:
            details.append(f"stderr:\n{stderr.strip()}" if stderr.strip() else "stderr: <empty>")
        super().__init__("\n".join(details))
        self.stdout = stdout
        self.stderr = stderr


class CLIAdapter(ChatAdapter):
    """Adapter that routes DSPy prompts through an arbitrary CLI process."""

    def __init__(
        self,
        cli_command: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        if isinstance(cli_command, str):
            raise ValueError("cli_command must be a sequence of strings, not a single string")
        if not cli_command:
            raise ValueError("cli_command cannot be empty")

        super().__init__(**kwargs)

        self.cli_command = list(cli_command)
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.encoding = encoding

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        formatted_inputs = dict(inputs)
        messages = self.format(processed_signature, demos, formatted_inputs)
        payload_text = self._build_payload(processed_signature, messages, demos, formatted_inputs, lm_kwargs)

        stdout = self._run_cli(payload_text)
        cli_outputs = self._parse_cli_outputs(stdout)
        return self._call_postprocess(processed_signature, signature, cli_outputs, lm)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        formatted_inputs = dict(inputs)
        messages = self.format(processed_signature, demos, formatted_inputs)
        payload_text = self._build_payload(processed_signature, messages, demos, formatted_inputs, lm_kwargs)

        stdout = await self._run_cli_async(payload_text)
        cli_outputs = self._parse_cli_outputs(stdout)
        return self._call_postprocess(processed_signature, signature, cli_outputs, lm)

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        processed_signature: type[Signature],
        messages: list[dict[str, Any]],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        lm_kwargs: dict[str, Any],
    ) -> str:
        payload = {
            "adapter": self.__class__.__name__,
            "messages": messages,
            "inputs": inputs,
            "demos": demos,
            "lm_kwargs": dict(lm_kwargs),
            "signature": self._serialize_signature(processed_signature),
        }
        # Ensure dumps always succeeds even if inputs contain non-JSON types.
        return json.dumps(payload, default=self._default_json_serializer, ensure_ascii=False)

    @staticmethod
    def _default_json_serializer(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()  # type: ignore[return-value]
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return repr(value)

    def _serialize_signature(self, signature: type[Signature]) -> dict[str, Any]:
        return {
            "name": signature.__name__,
            "instructions": getattr(signature, "instructions", ""),
            "input_fields": {
                name: self._serialize_field(field) for name, field in signature.input_fields.items()
            },
            "output_fields": {
                name: self._serialize_field(field) for name, field in signature.output_fields.items()
            },
        }

    @staticmethod
    def _serialize_field(field_info) -> dict[str, Any]:
        annotation = getattr(field_info, "annotation", None)
        try:
            annotation_name = get_annotation_name(annotation) if annotation is not None else None
        except Exception:  # pragma: no cover - best effort formatting
            annotation_name = repr(annotation)

        default = getattr(field_info, "default", None)
        if default is Ellipsis:  # pragma: no cover - defensive
            default = None

        return {
            "type": annotation_name,
            "description": getattr(field_info, "description", None),
            "default": default,
        }

    # ------------------------------------------------------------------
    # CLI execution helpers
    # ------------------------------------------------------------------
    def _run_cli(self, payload_text: str) -> str:
        try:
            completed = subprocess.run(  # noqa: S603
                self.cli_command,
                input=payload_text,
                capture_output=True,
                text=True,
                encoding=self.encoding,
                cwd=self.cwd,
                env=self._cli_env(),
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CLIAdapterError(f"CLI command not found: {self.cli_command}") from exc
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - relies on timing
            raise CLIAdapterError(
                f"CLI command '{self._command_display()}' timed out after {self.timeout} seconds",
                stdout=exc.stdout,
                stderr=exc.stderr,
            ) from exc

        if completed.returncode != 0:
            raise CLIAdapterError(
                f"CLI command '{self._command_display()}' exited with status {completed.returncode}",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        return completed.stdout

    async def _run_cli_async(self, payload_text: str) -> str:
        try:
            process = await asyncio.create_subprocess_exec(  # noqa: S603
                *self.cli_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self._cli_env(),
            )
        except FileNotFoundError as exc:
            raise CLIAdapterError(f"CLI command not found: {self.cli_command}") from exc

        input_bytes = payload_text.encode(self.encoding)
        communicate_coro = process.communicate(input_bytes)
        try:
            if self.timeout is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(communicate_coro, timeout=self.timeout)
            else:
                stdout_bytes, stderr_bytes = await communicate_coro
        except asyncio.TimeoutError as exc:  # pragma: no cover - relies on timing
            process.kill()
            await process.communicate()
            raise CLIAdapterError(
                f"CLI command '{self._command_display()}' timed out after {self.timeout} seconds"
            ) from exc

        stdout = stdout_bytes.decode(self.encoding, errors="replace")
        stderr = stderr_bytes.decode(self.encoding, errors="replace")

        if process.returncode != 0:
            raise CLIAdapterError(
                f"CLI command '{self._command_display()}' exited with status {process.returncode}",
                stdout=stdout,
                stderr=stderr,
            )

        return stdout

    def _parse_cli_outputs(self, stdout: str) -> list[Any]:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise CLIAdapterError("Could not decode CLI output as JSON", stdout=stdout) from exc

        outputs = parsed.get("outputs") if isinstance(parsed, dict) else parsed
        if not isinstance(outputs, list):
            raise CLIAdapterError("CLI output must be a list or dict with an 'outputs' field", stdout=stdout)

        return outputs

    def _cli_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env)
        return env

    def _command_display(self) -> str:
        try:
            return shlex.join(self.cli_command)
        except AttributeError:  # pragma: no cover - safety for older Python
            return " ".join(self.cli_command)
