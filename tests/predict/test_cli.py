from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

import dspy
from dspy.clients import cli_process
from dspy.clients.base_lm import BaseLM
from dspy.clients.cli_command import CLICommand
from dspy.core.types import LMImagePart, LMMessage, LMRequest, LMToolSpec
from dspy.clients.cli_stdio import _render_request
from dspy.predict.cli import CLI, CLIError
from dspy.utils.exceptions import AdapterParseError, LMUnsupportedFeatureError

PYTHON = sys.executable


def _python(source: str, *args: str) -> list[str]:
    return [PYTHON, "-c", source, *args]


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        return not stat.exists() or stat.read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def _wait_for_file(path: Path, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def _wait_until_dead(pid: int, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while _pid_is_alive(pid):
        if time.monotonic() >= deadline:
            raise AssertionError(f"process {pid} remained alive")
        time.sleep(0.01)


def _process_tree_command(pid_file: Path) -> list[str]:
    source = """
import pathlib
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    return _python(source, str(pid_file))


def _parallel_failure_command(pid_file: Path) -> list[str]:
    source = """
import os
import pathlib
import subprocess
import sys
import time

index = int(os.environ["DSPY_CLI_GENERATION_INDEX"])
if index == 0:
    deadline = time.monotonic() + 2
    while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    raise SystemExit(9)

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    return _python(source, str(pid_file))


class Payload(BaseModel):
    name: str
    scores: list[int]


class TypedResult(dspy.Signature):
    """Return a typed result."""

    task: str = dspy.InputField()
    count: int = dspy.OutputField()
    payload: Payload = dspy.OutputField()


class NestedItem(BaseModel):
    label: str
    values: dict[str, list[int]]


class NestedPayload(BaseModel):
    groups: list[NestedItem]


class NestedTypedResult(dspy.Signature):
    task: str = dspy.InputField()
    result: NestedPayload = dspy.OutputField()


class DefaultedResult(dspy.Signature):
    question: str = dspy.InputField()
    context: str = dspy.InputField(default="DEFAULT_CONTEXT")
    answer: str = dspy.OutputField()


class ImageResult(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField()


class EnclosingCLIProgram(dspy.Module):
    def __init__(self, command):
        super().__init__()
        self.cli = CLI(DefaultedResult, command=command)

    def forward(self, question: str):
        return self.cli(question=question)


class RejectLM(BaseLM):
    def __init__(self):
        super().__init__("reject")
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        raise AssertionError("the globally configured LM must not be called")


def test_requires_explicit_nonempty_argv():
    with pytest.raises(TypeError, match="argv sequence"):
        CLI("q -> a", command="echo hello")
    with pytest.raises(ValueError, match="cannot be empty"):
        CLI("q -> a", command=[])
    with pytest.raises(TypeError, match="non-empty string"):
        CLI("q -> a", command=["echo", ""])


def test_accepts_shared_command_spec():
    command = _python("print('[[ ## a ## ]]\\nok\\n[[ ## completed ## ]]')")
    cli = CLI("q -> a", command=CLICommand(argv=tuple(command)))

    assert cli(q="q").a == "ok"


def test_validates_execution_limits():
    with pytest.raises(ValueError, match="timeout"):
        CLI("q -> a", command=["echo"], timeout=0)
    with pytest.raises(ValueError, match="max_output_bytes"):
        CLI("q -> a", command=["echo"], max_output_bytes=0)


def test_rejects_tools_and_non_text_parts_before_execution():
    tool_request = LMRequest.from_call(model="cli/stdio", prompt="q", tools=[LMToolSpec(name="tool")])
    with pytest.raises(LMUnsupportedFeatureError, match="tools"):
        _render_request(tool_request)

    image_request = LMRequest(
        model="cli/stdio",
        messages=[LMMessage(role="user", parts=[LMImagePart(url="https://example.com/image.png")])],
    )
    with pytest.raises(LMUnsupportedFeatureError, match="image"):
        _render_request(image_request)


def test_exposes_one_ordinary_predict_node_with_the_original_signature():
    cli = CLI(TypedResult, command=["echo"])

    assert dict(cli.named_predictors()) == {"predict": cli.predict}
    assert cli.predict.lm is None
    assert set(cli.predict.signature.input_fields) == {"task"}
    assert set(cli.predict.signature.output_fields) == {"count", "payload"}
    assert "Return a typed result" in cli.predict.signature.instructions


def test_state_round_trip_excludes_the_command():
    source = "print('[[ ## a ## ]]\\nok\\n[[ ## completed ## ]]')"
    cli = CLI("q -> a", command=_python(source), timeout=3)
    cli.predict.demos = [dspy.Example(q="demo", a="ok").with_inputs("q")]
    state = cli.dump_state()

    assert "placeholder" not in str(state)
    assert PYTHON not in str(state)
    assert state["predict"]["lm"] is None

    target = CLI("q -> a", command=_python(source))
    target.load_state(state)

    assert len(target.predict.demos) == 1
    assert target._transport.command == tuple(_python(source))
    with dspy.context(lm=None):
        assert target(q="q").a == "ok"


def test_one_subprocess_no_configured_lm_and_typed_outputs(tmp_path: Path):
    calls_file = tmp_path / "calls"
    stdin_file = tmp_path / "stdin"
    source = """
import pathlib
import sys

calls = pathlib.Path(sys.argv[1])
calls.write_text(calls.read_text() + "call\\n" if calls.exists() else "call\\n")
pathlib.Path(sys.argv[2]).write_text(sys.stdin.read())
print('[[ ## count ## ]]')
print('42')
print('[[ ## payload ## ]]')
print('{"name": "result", "scores": [1, 2]}')
print('[[ ## completed ## ]]')
"""
    cli = CLI(TypedResult, command=_python(source, str(calls_file), str(stdin_file)))
    global_lm = RejectLM()

    with dspy.context(lm=global_lm, trace=[]):
        result = cli(task="one task")
        trace = list(dspy.settings.trace)

    assert global_lm.calls == 0
    assert calls_file.read_text().splitlines() == ["call"]
    rendered_stdin = stdin_file.read_text()
    assert "one task" in rendered_stdin
    assert "count" in rendered_stdin
    assert "payload" in rendered_stdin
    assert result.count == 42
    assert isinstance(result.count, int)
    assert result.payload == Payload(name="result", scores=[1, 2])
    assert set(result) == {"count", "payload"}
    assert len(trace) == 1
    assert trace[0][0] is cli.predict
    assert trace[0][1] == {"task": "one task"}
    assert trace[0][2] is result


def test_nested_typed_output_uses_normal_adapter_parsing():
    source = """
print('[[ ## result ## ]]')
print('{"groups": [{"label": "alpha", "values": {"scores": [1, 2]}}]}')
print('[[ ## completed ## ]]')
"""
    cli = CLI(NestedTypedResult, command=_python(source))

    result = cli(task="nested")

    assert result.result == NestedPayload(groups=[NestedItem(label="alpha", values={"scores": [1, 2]})])


def test_public_n_runs_independent_generations_in_order(tmp_path: Path):
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    source = """
import os
import pathlib
import sys

index = os.environ["DSPY_CLI_GENERATION_INDEX"]
total = os.environ["DSPY_CLI_TOTAL_GENERATIONS"]
pathlib.Path(sys.argv[1], index).touch()
print('[[ ## answer ## ]]')
print(f'{index}/{total}')
print('[[ ## completed ## ]]')
"""
    cli = CLI("q -> answer", command=_python(source, str(calls_dir)))

    result = cli(q="q", config={"n": 3})

    assert result.completions.answer == ["0/3", "1/3", "2/3"]
    assert sorted(path.name for path in calls_dir.iterdir()) == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_async_public_n_runs_independent_generations_in_order():
    source = """
import os

index = os.environ["DSPY_CLI_GENERATION_INDEX"]
total = os.environ["DSPY_CLI_TOTAL_GENERATIONS"]
print('[[ ## answer ## ]]')
print(f'{index}/{total}')
print('[[ ## completed ## ]]')
"""
    cli = CLI("q -> answer", command=_python(source))

    result = await cli.acall(q="q", config={"n": 3})

    assert result.completions.answer == ["0/3", "1/3", "2/3"]


def test_public_n_rejects_zero_before_process_spawn(tmp_path: Path):
    spawn_marker = tmp_path / "spawned"
    cli = CLI(
        "q -> answer", command=_python("import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", str(spawn_marker))
    )

    with pytest.raises(ValueError, match="n must be at least 1"):
        cli(q="q", config={"n": 0})

    assert not spawn_marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
def test_public_n_failure_kills_sibling_process_tree(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    cli = CLI("q -> answer", command=_parallel_failure_command(pid_file))

    with pytest.raises(CLIError, match="status 9"):
        cli(q="q", config={"n": 2})

    _wait_for_file(pid_file)
    _wait_until_dead(int(pid_file.read_text()))


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
async def test_async_public_n_failure_kills_sibling_process_tree(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    cli = CLI("q -> answer", command=_parallel_failure_command(pid_file))

    with pytest.raises(CLIError, match="status 9"):
        await cli.acall(q="q", config={"n": 2})

    await asyncio.to_thread(_wait_for_file, pid_file)
    await asyncio.to_thread(_wait_until_dead, int(pid_file.read_text()))


def test_enclosing_module_state_round_trip_reconstructs_from_init():
    source = "print('[[ ## answer ## ]]\\nok\\n[[ ## completed ## ]]')"
    program = EnclosingCLIProgram(_python(source))
    state = program.dump_state()

    assert set(state) == {"cli.predict"}
    assert state["cli.predict"]["lm"] is None

    target = EnclosingCLIProgram(_python(source))
    target.load_state(state)

    assert target.cli._transport.command == tuple(_python(source))
    assert target(question="q").answer == "ok"


def test_optimizer_preserves_transport_discovers_predictor_defaults_and_trace(tmp_path: Path):
    stdin_file = tmp_path / "stdin"
    source = """
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())
print('[[ ## answer ## ]]')
print('ok')
print('[[ ## completed ## ]]')
"""
    program = EnclosingCLIProgram(_python(source, str(stdin_file)))
    trainset = [dspy.Example(question="train", context="train context", answer="ok").with_inputs("question", "context")]

    compiled = dspy.LabeledFewShot(k=1).compile(program, trainset=trainset, sample=False)
    global_lm = RejectLM()
    with dspy.context(lm=global_lm, trace=[]):
        result = compiled(question="live")
        trace = list(dspy.settings.trace)

    assert [name for name, _ in compiled.named_predictors()] == ["cli.predict"]
    assert len(compiled.cli.predict.demos) == 1
    assert global_lm.calls == 0
    assert "DEFAULT_CONTEXT" in stdin_file.read_text()
    assert result.answer == "ok"
    assert len(trace) == 1
    assert trace[0][0] is compiled.cli.predict
    assert trace[0][1] == {"question": "live", "context": "DEFAULT_CONTEXT"}


def test_output_names_can_collide_with_prediction_methods():
    command = _python('print(\'[[ ## labels ## ]]\\n[\\"a\\", \\"b\\"]\\n[[ ## completed ## ]]\')')
    cli = CLI("question -> labels: list[str]", command=command)

    with dspy.context(lm=None):
        result = cli(question="q")

    assert result["labels"] == ["a", "b"]


def test_nonzero_exit_reports_bounded_stderr():
    cli = CLI("q -> a", command=_python("import sys; sys.stderr.write('diagnostic'); raise SystemExit(7)"))

    with pytest.raises(CLIError, match="status 7") as exc_info:
        cli._transport._invoke("prompt")

    assert exc_info.value.returncode == 7
    assert exc_info.value.stderr == "diagnostic"
    assert "stderr:\ndiagnostic" in str(exc_info.value)


def test_public_failure_runs_the_command_once(tmp_path: Path):
    calls_file = tmp_path / "calls"
    source = """
import pathlib
import sys

calls = pathlib.Path(sys.argv[1])
calls.write_text(calls.read_text() + "call\\n" if calls.exists() else "call\\n")
sys.stderr.write("failed")
raise SystemExit(3)
"""
    cli = CLI("q -> a", command=_python(source, str(calls_file)))

    with pytest.raises(CLIError, match="status 3"):
        cli(q="question")

    assert calls_file.read_text().splitlines() == ["call"]


def test_malformed_success_runs_the_command_once(tmp_path: Path):
    calls_file = tmp_path / "calls"
    source = """
import pathlib
import sys

calls = pathlib.Path(sys.argv[1])
calls.write_text(calls.read_text() + "call\\n" if calls.exists() else "call\\n")
print("malformed output")
"""
    cli = CLI("q -> answer: int", command=_python(source, str(calls_file)))

    with pytest.raises(AdapterParseError):
        cli(q="question")

    assert calls_file.read_text().splitlines() == ["call"]


@pytest.mark.asyncio
async def test_async_public_failure_runs_the_command_once(tmp_path: Path):
    calls_file = tmp_path / "calls"
    source = """
import pathlib
import sys

calls = pathlib.Path(sys.argv[1])
calls.write_text(calls.read_text() + "call\\n" if calls.exists() else "call\\n")
sys.stderr.write("failed")
raise SystemExit(4)
"""
    cli = CLI("q -> a", command=_python(source, str(calls_file)))

    with pytest.raises(CLIError, match="status 4"):
        await cli.acall(q="question")

    assert calls_file.read_text().splitlines() == ["call"]


def test_unsupported_tools_are_rejected_before_process_spawn(tmp_path: Path):
    spawn_marker = tmp_path / "spawned"
    cli = CLI("q -> a", command=_python("import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", str(spawn_marker)))
    request = LMRequest.from_call(model="cli/stdio", prompt="q", tools=[LMToolSpec(name="tool")])

    with pytest.raises(LMUnsupportedFeatureError, match="tools"):
        cli._transport.forward(request)

    assert not spawn_marker.exists()


def test_public_non_text_input_is_rejected_before_process_spawn(tmp_path: Path):
    spawn_marker = tmp_path / "spawned"
    cli = CLI(
        ImageResult,
        command=_python("import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", str(spawn_marker)),
    )

    with pytest.raises(LMUnsupportedFeatureError, match="image"):
        cli(image=dspy.Image("https://example.com/image.png"))

    assert not spawn_marker.exists()


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_sync_capture_is_bounded(stream: str):
    target = "sys.stdout" if stream == "stdout" else "sys.stderr"
    cli = CLI("q -> a", command=_python(f"import sys; {target}.write('x' * 10000)"), max_output_bytes=100)

    with pytest.raises(CLIError, match="exceeded max_output_bytes") as exc_info:
        cli._transport._invoke("prompt")

    assert len(getattr(exc_info.value, stream).encode()) == 100


def test_sync_timeout_kills_the_process_tree(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    cli = CLI("q -> a", command=_process_tree_command(pid_file), timeout=0.2)

    with pytest.raises(CLIError, match="timed out"):
        cli._transport._invoke("prompt")

    _wait_for_file(pid_file)
    child_pid = int(pid_file.read_text())
    _wait_until_dead(child_pid)


@pytest.mark.asyncio
async def test_aforward_pipes_stdin_and_extracts_output():
    command = _python("print('[[ ## answer ## ]]\\nHELLO\\n[[ ## completed ## ]]')")
    cli = CLI("question -> answer", command=command)

    with dspy.context(lm=None):
        result = await cli.acall(question="q")

    assert result.answer == "HELLO"


@pytest.mark.asyncio
async def test_async_timeout_includes_blocked_stdin():
    cli = CLI(
        "q -> a",
        command=_python("import time; time.sleep(60)"),
        timeout=0.1,
    )

    with pytest.raises(CLIError, match="timed out"):
        await cli._transport._ainvoke("x" * 10_000_000)


@pytest.mark.asyncio
async def test_async_cancellation_kills_the_process_tree(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    cli = CLI("q -> a", command=_process_tree_command(pid_file))
    invocation = asyncio.create_task(cli._transport._ainvoke("prompt"))
    await asyncio.to_thread(_wait_for_file, pid_file)
    child_pid = int(pid_file.read_text())

    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation

    await asyncio.to_thread(_wait_until_dead, child_pid)


@pytest.mark.asyncio
async def test_async_capture_is_bounded():
    cli = CLI(
        "q -> a",
        command=_python("import sys; sys.stdout.write('x' * 10000)"),
        max_output_bytes=100,
    )

    with pytest.raises(CLIError, match="exceeded max_output_bytes") as exc_info:
        await cli._transport._ainvoke("prompt")

    assert len(exc_info.value.stdout.encode()) == 100


def test_missing_command_is_a_structured_transport_error():
    cli = CLI("q -> a", command=["definitely-not-a-real-command-dspy-test"])

    with pytest.raises(CLIError, match="failed to start command") as exc_info:
        cli._transport._invoke("prompt")

    assert exc_info.value.model == "cli/stdio"
    assert exc_info.value.provider == "stdio"
    assert exc_info.value.returncode is None


def test_globally_configured_adapter_is_honored():
    source = """
print('{"a": "json-ok"}')
"""
    cli = CLI("q -> a", command=_python(source))

    with dspy.context(adapter=dspy.JSONAdapter(), lm=None):
        assert cli(q="q").a == "json-ok"


def test_preamble_chatter_and_trailing_signoff_are_tolerated():
    source = """
print("I'll help with that!")
print('[[ ## a ## ]]')
print('real answer')
print('[[ ## completed ## ]]')
print('Hope that helps!')
"""
    cli = CLI("q -> a", command=_python(source))

    assert cli(q="q").a == "real answer"


def test_prompt_echo_captures_the_first_field_occurrence():
    # Characterization: ChatAdapter takes the FIRST occurrence of each field
    # marker, so a command that echoes the rendered prompt before answering is
    # captured at the echo. The shipped presets keep stdout clean; arbitrary
    # commands must too.
    source = """
print('[[ ## a ## ]]')
print('ECHOED')
print('[[ ## completed ## ]]')
print('[[ ## a ## ]]')
print('real answer')
print('[[ ## completed ## ]]')
"""
    cli = CLI("q -> a", command=_python(source))

    assert cli(q="q").a == "ECHOED"


def test_user_env_overrides_generation_variables():
    source = """
import os
print('[[ ## a ## ]]')
print(os.environ['DSPY_CLI_GENERATION_INDEX'])
print('[[ ## completed ## ]]')
"""
    cli = CLI("q -> a", command=_python(source), env={"DSPY_CLI_GENERATION_INDEX": "7"})

    assert cli(q="q").a == "7"


@pytest.mark.skipif(os.name != "posix", reason="process-group escape is POSIX-specific")
def test_group_escaped_descendant_does_not_hang_cleanup(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_process, "_CLEANUP_GRACE_SECONDS", 0.5)
    pid_file = tmp_path / "escaped.pid"
    source = """
import pathlib
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
print('[[ ## a ## ]]')
print('ok')
print('[[ ## completed ## ]]')
"""
    cli = CLI("q -> a", command=_python(source, str(pid_file)), timeout=5)
    start = time.monotonic()
    try:
        assert cli(q="q").a == "ok"
        assert time.monotonic() - start < 4, "cleanup blocked on an escaped descendant's open pipe"
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
