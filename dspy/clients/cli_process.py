"""Internal process mechanics shared by CLI-backed DSPy abstractions."""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import BinaryIO

_READ_SIZE = 64 * 1024


@dataclass
class CLIProcessError(Exception):
    """A subprocess failed at the text stdin/stdout boundary."""

    message: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1

    def __str__(self) -> str:
        return self.message


class CLIProcess:
    """Run explicit argv once with bounded text stdin, stdout, and stderr."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        encoding: str = "utf-8",
        max_output_bytes: int = 1_000_000,
    ) -> None:
        if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
            raise TypeError("command must be an argv sequence, not a shell command string")
        if not command:
            raise ValueError("command cannot be empty")
        if any(not isinstance(argument, str) or not argument for argument in command):
            raise TypeError("every command argument must be a non-empty string")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero")
        codecs.lookup(encoding)

        self.command = tuple(command)
        self.cwd = os.fspath(cwd) if cwd is not None else None
        self.env = dict(env or {})
        self.timeout = timeout
        self.encoding = encoding
        self.max_output_bytes = max_output_bytes

    def run(
        self,
        prompt: str,
        *,
        generation_index: int = 0,
        total_generations: int = 1,
        cancel: threading.Event | None = None,
    ) -> str:
        input_bytes = self._encode(prompt)
        process = self._popen(generation_index, total_generations)
        stdout = bytearray()
        stderr = bytearray()
        overflow = threading.Event()
        readers = [
            threading.Thread(target=self._read_sync, args=(process.stdout, stdout, overflow), daemon=True),
            threading.Thread(target=self._read_sync, args=(process.stderr, stderr, overflow), daemon=True),
        ]
        writer = threading.Thread(target=self._write_sync, args=(process.stdin, input_bytes), daemon=True)
        for thread in (*readers, writer):
            thread.start()

        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        failure: str | None = None
        try:
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    failure = "CLI generation cancelled after a sibling failed"
                    break
                if overflow.is_set():
                    failure = self._limit_message()
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    failure = "CLI command timed out"
                    break
                try:
                    process.wait(timeout=0.02)
                except subprocess.TimeoutExpired:
                    pass
            self._kill_tree(process.pid)
            process.wait()
        except BaseException:
            self._kill_tree(process.pid)
            process.wait()
            raise
        finally:
            for thread in (*readers, writer):
                thread.join()

        if failure is None and overflow.is_set():
            failure = self._limit_message()
        return self._finish(process.returncode, stdout, stderr, failure)

    async def arun(
        self,
        prompt: str,
        *,
        generation_index: int = 0,
        total_generations: int = 1,
    ) -> str:
        input_bytes = self._encode(prompt)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **self._process_kwargs(generation_index, total_generations),
            )
        except OSError as exc:
            raise CLIProcessError(f"Could not start CLI command: {exc}") from exc

        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._kill_tree(process.pid)
            await process.wait()
            raise CLIProcessError("CLI subprocess pipes were not created", returncode=process.returncode or -1)

        stdout = bytearray()
        stderr = bytearray()
        overflow = asyncio.Event()
        readers = [
            asyncio.create_task(self._read_async(process.stdout, stdout, overflow)),
            asyncio.create_task(self._read_async(process.stderr, stderr, overflow)),
        ]
        writer = asyncio.create_task(self._write_async(process.stdin, input_bytes))

        async def wait_and_cleanup_descendants() -> None:
            # asyncio's Process.wait() may wait for pipe EOF after the direct
            # child exits. Descendants can retain those pipes, so observe the
            # returncode first and terminate the process group before waiting.
            while process.returncode is None:
                await asyncio.sleep(0.01)
            self._kill_tree(process.pid)
            await process.wait()

        process_wait = asyncio.create_task(wait_and_cleanup_descendants())
        overflow_wait = asyncio.create_task(overflow.wait())
        failure: str | None = None
        try:
            done, _ = await asyncio.wait(
                {process_wait, overflow_wait}, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                failure = "CLI command timed out"
            elif overflow_wait in done and overflow.is_set():
                failure = self._limit_message()
            self._kill_tree(process.pid)
            await process_wait
            await asyncio.gather(writer, *readers)
            if failure is None and overflow.is_set():
                failure = self._limit_message()
        except BaseException:
            self._kill_tree(process.pid)
            await process.wait()
            raise
        finally:
            overflow_wait.cancel()
            await asyncio.gather(overflow_wait, return_exceptions=True)
            for task in (writer, *readers):
                if not task.done():
                    task.cancel()
            await asyncio.gather(writer, *readers, return_exceptions=True)

        return self._finish(process.returncode, stdout, stderr, failure)

    def _encode(self, prompt: str) -> bytes:
        try:
            return prompt.encode(self.encoding)
        except UnicodeError as exc:
            raise CLIProcessError(f"CLI input is not valid {self.encoding}: {exc}") from exc

    def _popen(self, generation_index: int, total: int) -> subprocess.Popen[bytes]:
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **self._process_kwargs(generation_index, total),
            )
        except OSError as exc:
            raise CLIProcessError(f"Could not start CLI command: {exc}") from exc
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._kill_tree(process.pid)
            process.wait()
            raise CLIProcessError("CLI subprocess pipes were not created", returncode=process.returncode or -1)
        return process

    def _process_kwargs(self, generation_index: int, total: int) -> dict:
        environment = os.environ.copy()
        environment.update(self.env)
        environment["DSPY_CLI_GENERATION_INDEX"] = str(generation_index)
        environment["DSPY_CLI_TOTAL_GENERATIONS"] = str(total)
        kwargs = {"cwd": self.cwd, "env": environment}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        return kwargs

    def _read_sync(self, stream: BinaryIO | None, destination: bytearray, overflow: threading.Event) -> None:
        if stream is None:
            overflow.set()
            return
        try:
            while chunk := stream.read(_READ_SIZE):
                remaining = self.max_output_bytes - len(destination)
                destination.extend(chunk[: max(remaining, 0)])
                if len(chunk) > remaining:
                    overflow.set()
                    return
        finally:
            stream.close()

    @staticmethod
    def _write_sync(stream: BinaryIO | None, content: bytes) -> None:
        if stream is None:
            return
        try:
            stream.write(content)
            stream.close()
        except (BrokenPipeError, OSError):
            pass

    async def _read_async(self, stream: asyncio.StreamReader, destination: bytearray, overflow: asyncio.Event) -> None:
        while chunk := await stream.read(_READ_SIZE):
            remaining = self.max_output_bytes - len(destination)
            destination.extend(chunk[: max(remaining, 0)])
            if len(chunk) > remaining:
                overflow.set()
                return

    @staticmethod
    async def _write_async(stream: asyncio.StreamWriter, content: bytes) -> None:
        try:
            stream.write(content)
            await stream.drain()
            stream.close()
            await stream.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _finish(self, returncode: int, stdout: bytearray, stderr: bytearray, failure: str | None) -> str:
        stdout_text = stdout.decode(self.encoding, errors="replace")
        stderr_text = stderr.decode(self.encoding, errors="replace")
        if failure is not None:
            raise CLIProcessError(failure, stdout_text, stderr_text, returncode)
        if returncode != 0:
            raise CLIProcessError("CLI command exited unsuccessfully", stdout_text, stderr_text, returncode)
        try:
            return bytes(stdout).decode(self.encoding)
        except UnicodeError as exc:
            raise CLIProcessError(
                f"CLI stdout is not valid {self.encoding}: {exc}", stdout_text, stderr_text, returncode
            ) from exc

    def _limit_message(self) -> str:
        return f"CLI output limit exceeded ({self.max_output_bytes} bytes per stream)"

    @staticmethod
    def _kill_tree(pid: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(pid, signal.SIGKILL)
            else:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


__all__ = ["CLIProcess", "CLIProcessError"]
