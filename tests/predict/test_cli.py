import asyncio
import json
import os
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel

import dspy
from dspy.experimental import CLI, CLIError
from dspy.predict import cli as cli_module

pytestmark = pytest.mark.skipif(os.name != "posix", reason="CLI is POSIX only")

# What every stand-in agent starts with: it has read its task, and can hand over a result
# the way a real agent would, by running `submit` from its shell.
PRELUDE = """
import json, os, pathlib, subprocess, sys
task = sys.stdin.read()

def submit(value, stdin=False):
    text = value if isinstance(value, str) else json.dumps(value)
    run = subprocess.run(["submit"] if stdin else ["submit", text], input=text if stdin else None,
                         capture_output=True, text=True)
    print(run.returncode, run.stdout.strip() or run.stderr.strip())
"""


def agent(body: str, *args: object) -> list[str]:
    return [sys.executable, "-c", PRELUDE + textwrap.dedent(body), *map(str, args)]


def shell(script: str, *args: object) -> list[str]:
    """A stand-in that starts in milliseconds, for tests about time."""
    return ["sh", "-c", script, *map(str, args)]


def wait_for(condition, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    return condition()


def written(path: Path) -> bool:
    return path.exists() and path.read_text().endswith("\n")


def alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def workspaces(tmp_path: Path, monkeypatch) -> Path:
    """Where each call's private directory is made, so that a test can see it is gone again."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class Finding(BaseModel):
    file: str
    line: int


class Review(dspy.Signature):
    """Review the change and report what is wrong with it."""

    diff: str = dspy.InputField(desc="a unified diff")
    limits: dict[str, int] = dspy.InputField()
    verdict: str = dspy.OutputField(desc="approve or reject")
    count: int = dspy.OutputField()
    findings: list[Finding] = dspy.OutputField()


ANSWER = {"verdict": "reject", "count": 1, "findings": [{"file": "a.py", "line": 3}]}
INPUTS = {"diff": "- old\n+ new", "limits": {"max_findings": 5}}


def test_completes_a_signature_with_typed_outputs(workspaces: Path):
    result = CLI(Review, agent(f"print('looked at it'); submit({ANSWER!r})"))(**INPUTS)

    assert (result.verdict, result.count, result.findings) == ("reject", 1, [Finding(file="a.py", line=3)])
    assert result.transcript.splitlines() == ["looked at it", "0 Accepted. Your task is complete: stop now."]
    assert list(workspaces.iterdir()) == []
    # A field may be called anything, including what `Prediction` calls its own arguments.
    assert CLI("q -> base", agent("submit({'base': 'kept'})"))(q="q").base == "kept"


def test_the_task_states_what_to_do_with_what_and_how_to_hand_it_back(tmp_path: Path):
    seen = tmp_path / "task"
    CLI(Review, agent(f"pathlib.Path(sys.argv[1]).write_text(task); submit({ANSWER!r})", seen))(**INPUTS)

    task = seen.read_text()
    assert task.startswith(
        "Review the change and report what is wrong with it.\n\n# Inputs\n## diff (a unified diff)\n- old\n+ new"
    )
    assert '## limits\n{\n  "max_findings": 5\n}' in task
    assert "submit '<JSON object>'" in task and "[[ ##" not in task
    schema = json.loads(task.split("JSON Schema:\n\n")[1].split("\n\nIf submit")[0])
    assert schema["required"] == ["verdict", "count", "findings"] and schema["additionalProperties"] is False
    assert schema["properties"]["verdict"]["description"] == "approve or reject"
    assert schema["properties"]["findings"]["items"] == {"$ref": "#/$defs/Finding"} and "Finding" in schema["$defs"]


def test_a_rejected_submission_is_explained_and_can_be_corrected():
    body = f"""
        submit("{{not json")
        submit("[1, 2]")
        submit({{"verdict": "reject", "count": "several", "findings": [], "extra": 1}})
        submit({ANSWER!r}, stdin=True)
        submit({{**{ANSWER!r}, "verdict": "approve"}})
    """
    result = CLI(Review, agent(body))(**INPUTS)

    lines = result.transcript.splitlines()
    assert lines[0].startswith("1 Rejected: submission: Invalid JSON")
    assert lines[1].startswith("1 Rejected: submission: Input should be an object")
    assert lines[2].startswith("1 Rejected: ") and lines[2].endswith(". Fix the JSON object and run submit again.")
    assert "count: Input should be a valid integer" in lines[2] and "extra: Extra inputs are not permitted" in lines[2]
    assert lines[3:] == [
        "0 Accepted. Your task is complete: stop now.",
        "1 A result was already accepted. There is nothing more to submit: stop.",
    ]
    assert result.verdict == "reject"


def test_a_command_that_never_submits_is_an_error(workspaces: Path):
    with pytest.raises(CLIError, match="command ended without an accepted submission") as caught:
        CLI(Review, agent("print('the answer is reject')"))(**INPUTS)
    assert (caught.value.transcript, caught.value.returncode, caught.value.rejected) == (
        "the answer is reject\n",
        0,
        [],
    )

    body = """
        submit({"verdict": "reject"})
        sys.stderr.write("y" * 5000 + "gave up")
        sys.exit(3)
    """
    with pytest.raises(CLIError, match="command exited with status 3 without an accepted submission") as caught:
        CLI(Review, agent(body))(**INPUTS)
    assert caught.value.rejected == ["count: Field required; findings: Field required"]
    assert "rejected submission: count: Field required" in str(caught.value)
    assert str(caught.value).endswith("gave up") and len(str(caught.value)) < 2500 < len(caught.value.stderr)

    with pytest.raises(CLIError, match="failed to start command") as caught:
        CLI(Review, ["dspy-cli-no-such-program"])(**INPUTS)
    assert caught.value.returncode is None
    assert list(workspaces.iterdir()) == []


def test_an_accepted_submission_stands_whatever_the_command_does_next():
    assert CLI(Review, agent(f"submit({ANSWER!r}); sys.exit(5)"))(**INPUTS).verdict == "reject"


def test_the_call_ends_soon_after_acceptance_and_takes_the_command_with_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_module, "_SETTLE_SECONDS", 0.2)
    pid_file = tmp_path / "pid"
    script = """sleep 60 & echo $! > "$0"; submit '{"a": "x"}'; sleep 60"""

    started = time.monotonic()
    result = CLI("q -> a", shell(script, pid_file), timeout=None)(q="q")

    assert result.a == "x" and time.monotonic() - started < 30
    assert wait_for(lambda: not alive(pid_file))


def test_a_timeout_is_an_error_and_leaves_nothing_behind(workspaces: Path):
    with pytest.raises(CLIError, match=r"command timed out after 0\.2 seconds"):
        CLI("q -> a", ["sleep", "60"], timeout=0.2)(q="q")

    assert list(workspaces.iterdir()) == []


async def cancel_once_ready(script: str, ready: Path, *args: object) -> None:
    """Start a call, wait until its command says it is under way, then cancel the call."""
    call = asyncio.create_task(CLI("q -> a", shell(script, ready, *args)).acall(q="q"))
    assert await asyncio.to_thread(wait_for, lambda: written(ready))
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_a_stopped_command_is_asked_first_and_takes_what_it_started_with_it(tmp_path: Path, workspaces: Path):
    child, marker = tmp_path / "child", tmp_path / "marker"
    script = """trap 'echo cleaned up > "$1"; exit 0' TERM; sleep 60 & echo $! > "$0"; wait"""

    await cancel_once_ready(script, child, marker)

    assert await asyncio.to_thread(wait_for, lambda: written(marker)) and marker.read_text() == "cleaned up\n"
    assert await asyncio.to_thread(wait_for, lambda: not alive(child))
    assert await asyncio.to_thread(wait_for, lambda: not list(workspaces.glob("dspy-cli-*")))


@pytest.mark.asyncio
async def test_a_command_that_ignores_being_asked_is_killed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_module, "_STOP_SECONDS", 0.2)
    pid_file = tmp_path / "pid"

    await cancel_once_ready("""trap '' TERM; echo $$ > "$0"; exec sleep 60""", pid_file)

    assert await asyncio.to_thread(wait_for, lambda: not alive(pid_file))


def test_a_talkative_command_is_not_blocked_and_the_end_of_its_output_is_kept(monkeypatch):
    # An agent's log easily runs to megabytes. The result does not travel in it.
    monkeypatch.setattr(cli_module, "_KEEP_BYTES", 1000)
    body = "sys.stdout.write('x' * 300_000); sys.stderr.write('y' * 300_000); sys.stdout.flush(); submit({'a': 'x'})"

    transcript = CLI("q -> a", agent(body))(q="q").transcript

    assert len(transcript) == 1000 and transcript.endswith("0 Accepted. Your task is complete: stop now.\n")


def test_inputs_are_checked_before_anything_runs(tmp_path: Path):
    class Ask(dspy.Signature):
        question: str = dspy.InputField()
        style: str = dspy.InputField(default="terse")
        answer: str = dspy.OutputField()

    seen = tmp_path / "task"
    ask = CLI(Ask, agent("pathlib.Path(sys.argv[1]).write_text(task); submit({'answer': 'a'})", seen))

    with pytest.raises(ValueError, match="missing input fields: question"):
        ask(style="long")
    with pytest.raises(ValueError, match="not input fields of the signature: reviewer"):
        ask(question="q", reviewer="me")
    assert not seen.exists()

    ask(question="q")
    assert "## style\nterse" in seen.read_text()


def test_the_command_is_an_argv():
    with pytest.raises(TypeError, match="not a shell string"):
        CLI("q -> a", "claude -p")
    with pytest.raises(TypeError, match="non-empty argv"):
        CLI("q -> a", [])
    with pytest.raises(ValueError, match="'transcript' is what the command printed"):
        CLI("q -> transcript", ["agent"])
    # An empty argument is a legitimate one: some agents take `--tools ""` to mean none.
    assert CLI("q -> a", ["agent", "--tools", ""]).command == ["agent", "--tools", ""]


def test_it_runs_where_and_with_what_it_is_told(tmp_path: Path):
    ask = CLI(
        "q -> a",
        agent("submit({'a': os.getcwd() + ' ' + os.environ['GREETING']})"),
        cwd=tmp_path,
        env={"GREETING": "hi"},
    )

    assert ask(q="q").a == f"{tmp_path.resolve()} hi"


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_mix():
    echo = CLI("q -> a", agent("submit({'a': task.split('## q')[1].split()[0]})"))

    results = await asyncio.gather(*(echo.acall(q=f"question-{number}") for number in range(6)))

    assert [result.a for result in results] == [f"question-{number}" for number in range(6)]


def test_it_is_a_predictor_that_unmodified_dspy_code_can_evaluate_compile_save_and_load(tmp_path: Path):
    class Triage(dspy.Module):
        def __init__(self, command):
            super().__init__()
            self.review = CLI(Review, command)

        def forward(self, **inputs):
            return self.review(**inputs)

    seen = tmp_path / "task"
    program = Triage(agent(f"pathlib.Path(sys.argv[1]).write_text(task); submit({ANSWER!r})", seen))
    trainset = [dspy.Example(**INPUTS, verdict="reject").with_inputs("diff", "limits")]

    def metric(example, prediction, trace=None):
        return example.verdict == prediction.verdict

    assert [name for name, _ in program.named_predictors()] == ["review"]
    assert (
        dspy.Evaluate(devset=trainset * 4, metric=metric, num_threads=4, display_progress=False)(program).score == 100
    )

    compiled = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=1, max_labeled_demos=0).compile(
        program, trainset=trainset
    )
    # A demo is the call's inputs and outputs. What the command printed is not part of it.
    assert set(compiled.review.demos[0].keys()) == {"augmented", "diff", "limits", "verdict", "count", "findings"}

    compiled.review.signature = compiled.review.signature.with_instructions("Be strict about tests.")
    path = tmp_path / "program.json"
    compiled.save(path)
    assert sys.executable not in path.read_text()

    loaded = Triage(agent(f"pathlib.Path(sys.argv[1]).write_text(task); submit({ANSWER!r})", seen))
    loaded.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded(**INPUTS)
    # What an optimizer changes on a predictor is what the command is given.
    assert seen.read_text().startswith(
        'Be strict about tests.\n\n# Example 1\n## diff\n- old\n+ new\n## limits\n{\n  "max_findings": 5\n}\n## submitted\n{\n  "verdict": "reject",'
    )
