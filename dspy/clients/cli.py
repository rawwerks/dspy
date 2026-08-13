"""Run an arbitrary command as a DSPy language model over standard I/O."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from dspy.clients.base_lm import BaseLM
from dspy.clients.cli_command import CLICommand
from dspy.clients.cli_process import CLIProcess, CLIProcessError
from dspy.core.types import LMOutput, LMRequest, LMResponse, LMTextPart
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import LMTransportError, LMUnsupportedFeatureError


class CLIError(LMTransportError):
    """A command failed to satisfy the CLI transport contract."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        stdout: str = "",
        stderr: str = "",
        returncode: int = -1,
    ) -> None:
        details = [message]
        if stderr:
            details.append(f"stderr:\n{stderr}")
        super().__init__("\n".join(details), model=model, provider="cli", provider_code=str(returncode))
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class CLI(BaseLM):
    """Treat a command's stdin and stdout as a DSPy language model.

    DSPy's existing adapter renders the signature into text, this class sends
    that text to the command's standard input, and the adapter parses standard
    output into typed signature fields. Sampling flags, permissions, sessions,
    and other program-specific policy remain explicit command arguments.

    Example:
        ``dspy.CLI(dspy.resolve_agent_cli("codex", prefix_args=("-a", "never")))``

    Warning:
        The command runs on the host with the current process's authority. This
        class is a transport boundary, not a sandbox.
    """

    forward_contract = "typed_lm"

    def __init__(
        self,
        command: Sequence[str] | CLICommand,
        *,
        model: str = "cli",
        cwd: str | os.PathLike[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        encoding: str = "utf-8",
        max_output_bytes: int = 1_000_000,
        cache: bool = False,
        callbacks: list[BaseCallback] | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("model_type", None)
        kwargs.pop("num_retries", None)
        if isinstance(command, CLICommand):
            if command.prompt_transport != "stdin" or command.output_format != "text":
                raise ValueError("CLI currently supports only text over stdin/stdout")
            command = command.argv

        self._process = CLIProcess(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            encoding=encoding,
            max_output_bytes=max_output_bytes,
        )
        super().__init__(
            model=model,
            model_type="chat",
            cache=cache,
            callbacks=callbacks,
            num_retries=0,
            **kwargs,
        )
        self.command = list(self._process.command)
        self.cwd = self._process.cwd
        self.env = dict(self._process.env)
        self.timeout = self._process.timeout
        self.encoding = self._process.encoding
        self.max_output_bytes = self._process.max_output_bytes

    def forward(self, request: LMRequest) -> LMResponse:
        prompt = self._render_request(request)
        count = self._generation_count(request)
        if count == 1:
            outputs = [self._invoke(prompt, generation_index=0, total=count)]
        else:
            cancel = threading.Event()
            with ThreadPoolExecutor(max_workers=count) as executor:
                futures: dict[Future[str], int] = {
                    executor.submit(self._invoke, prompt, generation_index=index, total=count, cancel=cancel): index
                    for index in range(count)
                }
                outputs = [""] * count
                try:
                    for future in as_completed(futures):
                        outputs[futures[future]] = future.result()
                except BaseException:
                    cancel.set()
                    for future in futures:
                        future.cancel()
                    raise
        return self._response(request, outputs)

    async def aforward(self, request: LMRequest) -> LMResponse:
        prompt = self._render_request(request)
        count = self._generation_count(request)
        tasks = [
            asyncio.create_task(self._ainvoke(prompt, generation_index=index, total=count)) for index in range(count)
        ]
        try:
            outputs = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return self._response(request, outputs)

    def _render_request(self, request: LMRequest) -> str:
        if request.tools:
            raise LMUnsupportedFeatureError(
                "CLI does not define a provider-neutral native tool protocol",
                model=self.model,
                provider="cli",
                features=["native_tools"],
            )
        messages = []
        for message in request.messages:
            if any(not isinstance(part, LMTextPart) for part in message.parts):
                raise LMUnsupportedFeatureError(
                    "CLI's standard-input protocol supports text message parts only",
                    model=self.model,
                    provider="cli",
                    features=["non_text_input"],
                )
            messages.append(f"{message.role.upper()}:\n{''.join(part.text for part in message.parts)}")
        return "\n\n".join(messages)

    @staticmethod
    def _generation_count(request: LMRequest) -> int:
        count = 1 if request.config.n is None else request.config.n
        if count < 1:
            raise ValueError("n must be at least 1")
        return count

    def _invoke(
        self,
        prompt: str,
        *,
        generation_index: int,
        total: int,
        cancel: threading.Event | None = None,
    ) -> str:
        try:
            return self._process.run(
                prompt,
                generation_index=generation_index,
                total_generations=total,
                cancel=cancel,
            )
        except CLIProcessError as exc:
            raise self._error(exc) from exc

    async def _ainvoke(self, prompt: str, *, generation_index: int, total: int) -> str:
        try:
            return await self._process.arun(
                prompt,
                generation_index=generation_index,
                total_generations=total,
            )
        except CLIProcessError as exc:
            raise self._error(exc) from exc

    def _error(self, exc: CLIProcessError) -> CLIError:
        return CLIError(
            exc.message,
            model=self.model,
            stdout=exc.stdout,
            stderr=exc.stderr,
            returncode=exc.returncode,
        )

    @staticmethod
    def _response(request: LMRequest, outputs: list[str]) -> LMResponse:
        return LMResponse(
            model=request.model,
            outputs=[LMOutput(parts=[LMTextPart(text=output)]) for output in outputs],
        )

    def dump_state(self) -> dict:
        """Return trusted reconstruction state without environment values."""
        state = super().dump_state()
        state.update(
            command=list(self.command),
            cwd=self.cwd,
            timeout=self.timeout,
            encoding=self.encoding,
            max_output_bytes=self.max_output_bytes,
        )
        return state


__all__ = ["CLI", "CLIError"]
