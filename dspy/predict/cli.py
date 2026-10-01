"""Complete a signature by running a command: a coding agent's CLI, or anything that behaves like one."""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic_core import PydanticUndefined, to_json

from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import DSPyError

_POLL_SECONDS = 0.05
# After its submission is accepted, a command may take this long to end its turn by itself.
_SETTLE_SECONDS = 10.0
# When stopped, a command has this long to stop what it started before it is killed.
_STOP_SECONDS = 5.0
# How much of the end of stdout, and of stderr, is read back.
_KEEP_BYTES = 1_000_000

# The program an agent runs to hand over its result. It only carries text to the calling
# process and reports the verdict; all checking happens there, against the signature.
_SUBMIT = """#!/bin/sh
cd "$(dirname "$0")" || exit 2
if [ $# -gt 0 ]; then printf '%s' "$1" > $$.part; else cat > $$.part; fi
mv $$.part $$.submission
tries=0
while [ $tries -lt 300 ]; do
    if [ -e $$.accepted ]; then cat $$.accepted; rm $$.accepted; exit 0; fi
    if [ -e $$.rejected ]; then cat $$.rejected; rm $$.rejected; exit 1; fi
    sleep 0.1
    tries=$((tries + 1))
done
echo "submit: nothing is listening for this result any more." >&2
exit 2
"""

_HOW_TO_ANSWER = """# How to answer
Do not answer in prose. Hand over the result by running this command in your shell:

    submit '<JSON object>'

`submit` is on your PATH (it is {submit}). With no argument it reads the object from standard input. \
The object must match this JSON Schema:

{schema}

If submit rejects the object, fix it and run submit again. Printing the object does not deliver it: \
the result counts only once submit has accepted it. Then stop."""


class CLIError(DSPyError):
    """The command did not produce an accepted submission.

    Attributes:
        transcript: The end of what the command printed to standard output.
        stderr: The end of what it printed to standard error.
        returncode: Its exit status, or None if it never started.
        rejected: Why each rejected submission was turned down, in order.
    """

    def __init__(
        self, message: str, *, transcript: str = "", stderr: str = "", returncode: int | None = None, rejected=()
    ):
        details = [message, *(f"rejected submission: {reason}" for reason in rejected)]
        if stderr.strip():
            details.append(f"stderr:\n{stderr.strip()[-2000:]}")
        super().__init__("\n".join(details))
        self.transcript, self.stderr, self.returncode, self.rejected = transcript, stderr, returncode, list(rejected)


@experimental
class CLI(Predict):
    """A predictor whose forward pass is a command, such as a coding agent run headless.

    The signature's instructions, demos and inputs are written to the command's standard
    input as a task. The command does its work and finishes by running `submit` with one
    JSON object holding the output fields. The object is validated against the output
    fields' types. A rejected submission is answered with the reason, so an agent can
    correct it and submit again in the same run; the first accepted one becomes the
    `Prediction`. Nothing is parsed out of prose, and no language model is configured.

    `CLI` is a `Predict`: optimizers find it with `named_predictors()` and tune its
    instructions and demos. The command is not part of saved state.

    Args:
        signature: The signature to complete, as a class or a string.
        command: The argv to run for each call. Not a shell string.
        cwd: The directory the command works in. Defaults to the current one.
        env: Environment variables added to the inherited environment.
        timeout: Seconds a call may take before the command is stopped. None waits forever.

    Example:
        ```python
        from dspy.experimental import CLI

        find = CLI(
            "question -> files: list[str]",
            ["claude", "-p", "--allowedTools", "Read,Grep,Glob,Bash(submit:*)"],
            cwd="docs",
        )
        find(question="Where is caching configured?").files
        ```

    Warning:
        The command runs with the authority of the current process. This is not a
        sandbox, and inputs reach an agent that may act on them: grant it only what the
        task needs, with the agent's own flags.
    """

    def __init__(
        self,
        signature: str | type[Signature],
        command: Sequence[str],
        *,
        cwd: str | os.PathLike | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 600.0,
    ):
        super().__init__(signature)
        if os.name != "posix":
            raise NotImplementedError("CLI needs a POSIX system: `submit` is a shell script.")
        if isinstance(command, str) or not command or not all(isinstance(argument, str) for argument in command):
            raise TypeError("command must be a non-empty argv such as ['claude', '-p'], not a shell string")
        if "transcript" in self.signature.output_fields:
            raise ValueError("'transcript' is what the command printed; name the output field something else")
        self.command, self.cwd, self.env, self.timeout = list(command), cwd, dict(env or {}), timeout

    def forward(self, **inputs: Any) -> Prediction:
        return self._run(inputs, threading.Event())

    async def aforward(self, **inputs: Any) -> Prediction:
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._run, inputs, stop)
        finally:
            stop.set()  # Reached early only on cancellation, which then stops the command.

    def _run(self, inputs: dict[str, Any], stop: threading.Event) -> Prediction:
        signature = self.signature
        inputs = _bind(signature, inputs)
        outputs = create_model(
            "Outputs",
            __config__=ConfigDict(extra="forbid"),
            **{
                name: (field.annotation, Field(description=_desc(field)))
                for name, field in signature.output_fields.items()
            },
        )
        # Everything one call exchanges with its command lives here, and goes when the call ends.
        home = Path(tempfile.mkdtemp(prefix="dspy-cli-"))
        try:
            submit = home / "submit"
            submit.write_text(_SUBMIT)
            submit.chmod(0o700)
            (home / "task").write_text(_task(signature, self.demos, inputs, outputs, submit), encoding="utf-8")
            env = {**os.environ, **self.env}
            env["PATH"] = f"{home}{os.pathsep}{env.get('PATH', '')}"
            accepted, rejected, problem = None, [], None

            with open(home / "task", "rb") as stdin, open(home / "out", "wb") as out, open(home / "err", "wb") as err:
                try:
                    # Its own process group, so that it can be stopped together with what it starts.
                    process = subprocess.Popen(
                        self.command, cwd=self.cwd, env=env, stdin=stdin, stdout=out, stderr=err, start_new_session=True
                    )
                except OSError as error:
                    raise CLIError(f"failed to start command: {error}") from error
                try:
                    deadline = None if self.timeout is None else time.monotonic() + self.timeout
                    while True:
                        running = process.poll() is None
                        # Also once more after the command has exited, for a submission made just before.
                        newly = _answer(home, outputs, accepted is not None, rejected)
                        if newly is not None:
                            # Its work is done: what remains is a moment to end its turn.
                            accepted, deadline = newly, time.monotonic() + _SETTLE_SECONDS
                        if not running:
                            break
                        if stop.is_set():
                            problem = "command was cancelled"
                            break
                        if deadline is not None and time.monotonic() >= deadline:
                            if accepted is None:
                                problem = f"command timed out after {self.timeout} seconds"
                            break
                        time.sleep(_POLL_SECONDS)
                finally:
                    _stop(process)

            transcript, stderr = _tail(home / "out"), _tail(home / "err")
        finally:
            shutil.rmtree(home, ignore_errors=True)

        if accepted is None or problem:
            how = "ended" if process.returncode == 0 else f"exited with status {process.returncode}"
            raise CLIError(
                problem or f"command {how} without an accepted submission",
                transcript=transcript,
                stderr=stderr,
                returncode=process.returncode,
                rejected=rejected,
            )
        values = {name: getattr(accepted, name) for name in signature.output_fields}
        if settings.trace is not None and settings.max_trace_size > 0:
            if len(settings.trace) >= settings.max_trace_size:
                settings.trace.pop(0)
            # Demos are built from the trace, so the transcript is kept out of it.
            settings.trace.append((self, inputs, Prediction(base=values)))
        return Prediction(base={**values, "transcript": transcript})


def _answer(home: Path, outputs: type[BaseModel], done: bool, rejected: list[str]) -> BaseModel | None:
    """Give every submission that has arrived its verdict. Returns one that was newly accepted."""
    accepted = None
    for path in home.glob("*.submission"):
        text = path.read_text(encoding="utf-8", errors="replace")
        path.unlink()
        verdict, message = "rejected", "A result was already accepted. There is nothing more to submit: stop."
        if not done and accepted is None:
            try:
                accepted = outputs.model_validate_json(text)
                verdict, message = "accepted", "Accepted. Your task is complete: stop now."
            except ValidationError as error:
                reason = "; ".join(
                    f"{'.'.join(map(str, problem['loc'])) or 'submission'}: {problem['msg']}"
                    for problem in error.errors()
                )
                rejected.append(reason)
                message = f"Rejected: {reason}. Fix the JSON object and run submit again."
        reply = path.with_suffix(".reply")
        reply.write_text(message + "\n", encoding="utf-8")
        reply.replace(path.with_suffix(f".{verdict}"))
    return accepted


def _stop(process: subprocess.Popen) -> None:
    """End the command and its process group: asked first, so an agent can stop what it started, then killed."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
            process.wait(timeout=_STOP_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _tail(path: Path) -> str:
    with open(path, "rb") as file:
        file.seek(max(path.stat().st_size - _KEEP_BYTES, 0))
        return file.read().decode("utf-8", errors="replace")


def _bind(signature: type[Signature], values: dict[str, Any]) -> dict[str, Any]:
    """The call's inputs in signature order, with defaults filled in, or an error before anything runs."""
    fields = signature.input_fields
    if unknown := sorted(values.keys() - fields.keys()):
        raise ValueError(f"not input fields of the signature: {', '.join(unknown)}")
    defaults = {name: field.default for name, field in fields.items() if field.default is not PydanticUndefined}
    bound = {**defaults, **values}
    if missing := [name for name in fields if name not in bound]:
        raise ValueError(f"missing input fields: {', '.join(missing)}")
    return {name: bound[name] for name in fields}


def _desc(field: Any) -> str | None:
    description = (field.json_schema_extra or {}).get("desc", "")
    # DSPy fills in "${name}" for a field that was given no description.
    return None if description.startswith("${") else description or None


def _task(
    signature: type[Signature], demos: list, inputs: dict[str, Any], outputs: type[BaseModel], submit: Path
) -> str:
    """The task as the command reads it: what to do, examples, with what, and how to hand the result back."""

    def section(name: str, value: Any) -> str:
        return f"## {name}\n{value if isinstance(value, str) else to_json(value, indent=2, fallback=str).decode()}"

    parts = [signature.instructions.strip()]
    for number, demo in enumerate(demos, 1):
        given = [section(name, demo[name]) for name in signature.input_fields if name in demo]
        submitted = {name: demo[name] for name in signature.output_fields if name in demo}
        parts.append("\n".join([f"# Example {number}", *given, section("submitted", submitted)]))
    given = []
    for name, field in signature.input_fields.items():
        description = _desc(field)
        given.append(section(f"{name} ({description})" if description else name, inputs[name]))
    parts.append("\n".join(["# Inputs", *given]))
    parts.append(_HOW_TO_ANSWER.format(submit=submit, schema=json.dumps(outputs.model_json_schema(), indent=2)))
    return "\n\n".join(parts) + "\n"
