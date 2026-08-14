"""Private stdio transport backing :class:`dspy.CLI`. Not public API."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from dspy.clients.base_lm import BaseLM
from dspy.clients.cli_process import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    CLIProcess,
    CLIProcessError,
)
from dspy.core.types import LMOutput, LMRequest, LMResponse, LMTextPart
from dspy.utils.exceptions import LMTransportError, LMUnsupportedFeatureError


class CLIError(LMTransportError):
    """A command could not produce a usable response."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "", returncode: int | None = None):
        details = [message]
        if stderr:
            details.append(f"stderr:\n{stderr}")
        provider_code = str(returncode) if returncode is not None else None
        super().__init__("\n".join(details), model="cli/stdio", provider="stdio", provider_code=provider_code)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _StdioCLI(BaseLM):
    """Private typed LM that runs one command per generation over stdio."""

    forward_contract = "typed_lm"

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
        encoding: str = "utf-8",
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self._process = CLIProcess(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            encoding=encoding,
            max_output_bytes=max_output_bytes,
        )
        super().__init__(model="cli/stdio", cache=False, num_retries=0)

    @property
    def command(self) -> tuple[str, ...]:
        return self._process.command

    @property
    def cwd(self) -> str | None:
        return self._process.cwd

    @property
    def env(self) -> dict[str, str]:
        return dict(self._process.env)

    @property
    def timeout(self) -> float | None:
        return self._process.timeout

    @property
    def encoding(self) -> str:
        return self._process.encoding

    @property
    def max_output_bytes(self) -> int:
        return self._process.max_output_bytes

    def forward(self, request: LMRequest) -> LMResponse:
        prompt = _render_request(request)
        count = _generation_count(request)
        if count == 1:
            outputs = [self._invoke(prompt)]
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
        return _response(request, outputs)

    async def aforward(self, request: LMRequest) -> LMResponse:
        prompt = _render_request(request)
        count = _generation_count(request)
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
        return _response(request, outputs)

    def _invoke(
        self,
        prompt: str,
        *,
        generation_index: int = 0,
        total: int = 1,
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
            raise self._cli_error(exc) from exc

    async def _ainvoke(self, prompt: str, *, generation_index: int = 0, total: int = 1) -> str:
        try:
            return await self._process.arun(
                prompt,
                generation_index=generation_index,
                total_generations=total,
            )
        except CLIProcessError as exc:
            raise self._cli_error(exc) from exc

    def _cli_error(self, exc: CLIProcessError) -> CLIError:
        message = exc.message
        if exc.code == "exit":
            message = f"command exited with status {exc.returncode}"
        elif exc.code == "limit":
            message = f"command output exceeded max_output_bytes={self.max_output_bytes}"
        elif exc.code == "spawn":
            message = message.replace("Could not start CLI command", "failed to start command", 1)
            return CLIError(message, stdout=exc.stdout, stderr=exc.stderr)
        return CLIError(message, stdout=exc.stdout, stderr=exc.stderr, returncode=exc.returncode)


def _render_request(request: LMRequest) -> str:
    if request.tools:
        raise LMUnsupportedFeatureError(
            "stdin transport does not support tools",
            features=["tools"],
            model=request.model,
            provider="stdio",
        )
    messages = []
    for message in request.messages:
        if any(not isinstance(part, LMTextPart) for part in message.parts):
            unsupported = next(part.type for part in message.parts if not isinstance(part, LMTextPart))
            raise LMUnsupportedFeatureError(
                f"stdin transport does not support LM content part {unsupported!r}",
                features=["non_text_input"],
                model=request.model,
                provider="stdio",
            )
        messages.append(f"{message.role.upper()}:\n{''.join(part.text for part in message.parts)}")
    return "\n\n".join(messages)


def _generation_count(request: LMRequest) -> int:
    count = 1 if request.config.n is None else request.config.n
    if count < 1:
        raise ValueError("n must be at least 1")
    return count


def _response(request: LMRequest, outputs: list[str]) -> LMResponse:
    return LMResponse(
        model=request.model,
        outputs=[LMOutput(parts=[LMTextPart(text=output)]) for output in outputs],
    )


__all__ = ["CLIError"]
