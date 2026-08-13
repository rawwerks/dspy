from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

import dspy
from dspy.clients.cli import CLI, CLIError
from dspy.clients.cli_command import CLICommand
from dspy.core.types import LMImagePart, LMMessage, LMToolSpec
from dspy.utils.exceptions import AdapterParseError, LMError, LMUnsupportedFeatureError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cli_echo.py"


def command() -> list[str]:
    return [sys.executable, str(SCRIPT)]


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    try:
        return not stat.exists() or stat.read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_until_stopped(pid: int, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not process_is_running(pid)


def request(prompt: str = "hello", *, n: int | None = None) -> dspy.LMRequest:
    kwargs = {"n": n} if n is not None else {}
    return dspy.LMRequest.from_call(model="cli", prompt=prompt, **kwargs)


def test_cli_is_a_typed_lm_transport():
    cli = CLI(command())

    assert isinstance(cli, dspy.BaseLM)
    assert cli.forward_contract == "typed_lm"
    assert cli.command == command()


def test_cli_accepts_shared_command_spec():
    cli = CLI(CLICommand(argv=tuple(command())))

    assert cli(request("from spec")).text == "USER:\nfrom spec"


def test_cli_validates_encoding_before_starting_command(tmp_path):
    marker = tmp_path / "started"
    with pytest.raises(LookupError):
        CLI(
            [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            encoding="definitely-not-an-encoding",
        )
    assert not marker.exists()


def test_cli_rejects_unencodable_input_before_starting_command(tmp_path):
    marker = tmp_path / "started"
    cli = CLI(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
        encoding="ascii",
    )

    with pytest.raises(CLIError, match="input is not valid ascii"):
        cli(request("café"))
    assert not marker.exists()


@pytest.mark.parametrize("invalid", [[], (), "python script.py", ["python", ""]])
def test_cli_requires_explicit_nonempty_argv(invalid):
    with pytest.raises((TypeError, ValueError)):
        CLI(invalid)


def test_forward_pipes_rendered_request_to_stdin():
    cli = CLI(command())

    response = cli(request("hello from stdin"))

    assert response.text == "USER:\nhello from stdin"
    assert len(cli.history) == 1


def test_default_adapter_handles_typed_multi_output_signature():
    class Capital(dspy.Signature):
        """Answer the geography question and assess confidence."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        confidence: float = dspy.OutputField()

    cli = CLI(command(), env={"CLI_MODE": "chat_fields"})
    predictor = dspy.Predict(Capital)

    with dspy.context(lm=cli):
        result = predictor(question="What is the capital of France?")

    assert result.answer == "Paris"
    assert result.confidence == 0.93
    assert isinstance(result.confidence, float)
    assert predictor.signature is Capital


def test_cli_keeps_predict_optimizer_and_trace_semantics():
    class Program(dspy.Module):
        def __init__(self):
            self.generate = dspy.Predict("question -> answer, confidence: float")

        def forward(self, question):
            return self.generate(question=question)

    program = Program()
    cli = CLI(command(), env={"CLI_MODE": "chat_fields"})

    with dspy.context(lm=cli, trace=[]):
        result = program(question="What is the capital of France?")
        trace = list(dspy.settings.trace)

    assert dict(program.named_predictors()) == {"generate": program.generate}
    assert result.answer == "Paris"
    assert result.confidence == 0.93
    assert len(trace) == 1
    assert trace[0][0] is program.generate
    assert trace[0][1] == {"question": "What is the capital of France?"}
    assert trace[0][2] is result


def test_signature_defaults_remain_owned_by_predict():
    class Capital(dspy.Signature):
        question: str = dspy.InputField(default="What is the capital of France?")
        answer: str = dspy.OutputField()
        confidence: float = dspy.OutputField()

    cli = CLI(command(), env={"CLI_MODE": "chat_fields"})

    with dspy.context(lm=cli):
        result = dspy.Predict(Capital)()

    assert result.answer == "Paris"
    assert result.confidence == 0.93


def test_n_runs_independent_cli_generations_in_order():
    cli = CLI(command(), env={"CLI_MODE": "generation_index"})

    response = cli(request("ignored", n=3))

    assert [output.text for output in response.outputs] == ["0/3", "1/3", "2/3"]


def test_n_rejects_zero_generations():
    cli = CLI(command())

    with pytest.raises(ValueError, match="n must be at least 1"):
        cli(request(n=0))


@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
def test_sync_n_failure_kills_sibling_process_group(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    cli = CLI(command(), env={"CLI_MODE": "parallel_fail_cleanup", "CLI_PID_FILE": str(pid_file)})

    with pytest.raises(CLIError, match="exited unsuccessfully"):
        cli(request(n=2))

    assert pid_file.exists()
    wait_until_stopped(int(pid_file.read_text()))


@pytest.mark.asyncio
async def test_async_forward_uses_same_stdio_contract():
    cli = CLI(command())

    response = await cli.acall(request("async hello"))

    assert response.text == "USER:\nasync hello"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
async def test_async_n_failure_kills_sibling_process_group(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    cli = CLI(command(), env={"CLI_MODE": "parallel_fail_cleanup", "CLI_PID_FILE": str(pid_file)})

    with pytest.raises(CLIError, match="exited unsuccessfully"):
        await cli.acall(request(n=2))

    assert pid_file.exists()
    await asyncio.to_thread(wait_until_stopped, int(pid_file.read_text()))


def test_nonzero_exit_is_an_lm_error_and_does_not_trigger_adapter_retry(tmp_path):
    count_file = tmp_path / "count"
    cli = CLI(command(), env={"CLI_MODE": "fail_counted", "CLI_COUNT_FILE": str(count_file)})
    predictor = dspy.Predict("question -> answer")

    with dspy.context(lm=cli), pytest.raises(CLIError) as exc_info:
        predictor(question="fail once")

    assert isinstance(exc_info.value, LMError)
    assert exc_info.value.returncode == 2
    assert "intentional failure" in exc_info.value.stderr
    assert count_file.read_text() == "1"


@pytest.mark.asyncio
async def test_async_nonzero_exit_does_not_trigger_adapter_retry(tmp_path):
    count_file = tmp_path / "count"
    cli = CLI(command(), env={"CLI_MODE": "fail_counted", "CLI_COUNT_FILE": str(count_file)})
    predictor = dspy.Predict("question -> answer")

    with dspy.context(lm=cli), pytest.raises(CLIError):
        await predictor.acall(question="fail once")

    assert count_file.read_text() == "1"


def test_invalid_stdout_is_an_lm_error_and_does_not_trigger_adapter_retry(tmp_path):
    count_file = tmp_path / "count"
    cli = CLI(command(), env={"CLI_MODE": "invalid_output_counted", "CLI_COUNT_FILE": str(count_file)})
    predictor = dspy.Predict("question -> answer")

    with dspy.context(lm=cli), pytest.raises(CLIError, match="not valid utf-8"):
        predictor(question="decode once")

    assert count_file.read_text() == "1"


def test_malformed_success_follows_configured_adapter_retry_policy(tmp_path):
    count_file = tmp_path / "count"
    source = """
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(str(int(path.read_text()) + 1 if path.exists() else 1))
print("malformed output")
"""
    cli = CLI([sys.executable, "-c", source, str(count_file)])
    predictor = dspy.Predict("question -> answer: int")

    with dspy.context(lm=cli), pytest.raises(AdapterParseError):
        predictor(question="default fallback")
    assert count_file.read_text() == "2"

    count_file.write_text("0")
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    with dspy.context(lm=cli, adapter=adapter), pytest.raises(AdapterParseError):
        predictor(question="at most once")
    assert count_file.read_text() == "1"


def test_rejects_tools_and_non_text_parts_before_starting_command(tmp_path):
    marker = tmp_path / "started"
    cli = CLI([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    tool_request = dspy.LMRequest.from_call(model="cli", prompt="q", tools=[LMToolSpec(name="tool")])
    with pytest.raises(LMUnsupportedFeatureError, match="native tool protocol"):
        cli(tool_request)
    image_request = dspy.LMRequest(
        model="cli",
        messages=[LMMessage(role="user", parts=[LMImagePart(url="https://example.com/image.png")])],
    )
    with pytest.raises(LMUnsupportedFeatureError, match="text message parts only"):
        cli(image_request)
    assert not marker.exists()


def test_output_limit_fails_closed():
    cli = CLI(command(), env={"CLI_MODE": "large_output"}, max_output_bytes=128)

    with pytest.raises(CLIError, match="output limit"):
        cli(request())


@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
def test_timeout_kills_the_process_group(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    cli = CLI(
        command(),
        env={"CLI_MODE": "spawn_descendant", "CLI_PID_FILE": str(pid_file)},
        timeout=0.2,
    )

    with pytest.raises(CLIError, match="timed out"):
        cli(request())

    descendant_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            running = process_is_running(descendant_pid)
        except PermissionError:
            running = True
        if not running:
            break
        time.sleep(0.02)
    else:
        os.kill(descendant_pid, signal.SIGKILL)
        pytest.fail("CLI timeout left a descendant process alive")


@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
def test_sync_nonzero_exit_kills_descendants_that_retain_pipes(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    cli = CLI(command(), env={"CLI_MODE": "spawn_descendant_then_fail", "CLI_PID_FILE": str(pid_file)})

    with pytest.raises(CLIError, match="exited unsuccessfully") as exc_info:
        cli(request())

    assert "parent failed" in exc_info.value.stderr
    wait_until_stopped(int(pid_file.read_text()))


def test_sync_timeout_includes_blocked_stdin():
    cli = CLI(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout=0.1,
    )

    with pytest.raises(CLIError, match="timed out"):
        cli(request("x" * 10_000_000))


@pytest.mark.asyncio
async def test_async_timeout_includes_blocked_stdin():
    cli = CLI(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout=0.1,
    )

    with pytest.raises(CLIError, match="timed out"):
        await cli.acall(request("x" * 10_000_000))


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
async def test_async_cancellation_kills_the_process_group(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    cli = CLI(command(), env={"CLI_MODE": "spawn_descendant", "CLI_PID_FILE": str(pid_file)})

    task = asyncio.create_task(cli.acall(request()))
    deadline = time.monotonic() + 2
    while not pid_file.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    descendant_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while process_is_running(descendant_pid) and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert not process_is_running(descendant_pid)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="process-group liveness probe is POSIX-specific")
async def test_async_nonzero_exit_kills_descendants_that_retain_pipes(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    cli = CLI(command(), env={"CLI_MODE": "spawn_descendant_then_fail", "CLI_PID_FILE": str(pid_file)})

    with pytest.raises(CLIError, match="exited unsuccessfully") as exc_info:
        await cli.acall(request())

    assert "parent failed" in exc_info.value.stderr
    await asyncio.to_thread(wait_until_stopped, int(pid_file.read_text()))


def test_dump_state_preserves_transport_config_without_environment_values():
    cli = CLI(
        command(),
        cwd=str(SCRIPT.parent),
        env={"CLI_CONFIG": "sentinel-value"},
        timeout=12,
        max_output_bytes=4096,
    )

    state = cli.dump_state()

    assert state["command"] == command()
    assert state["cwd"] == str(SCRIPT.parent)
    assert state["timeout"] == 12
    assert state["max_output_bytes"] == 4096
    assert "env" not in state
    assert "sentinel-value" not in json.dumps(state)

    with pytest.raises(ValueError, match="custom serialized LM class"):
        dspy.BaseLM.load_state(state)
    restored = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)
    assert isinstance(restored, CLI)
    assert restored.command == command()
    assert restored.env == {}
    assert restored(request("restored")).text == "USER:\nrestored"
