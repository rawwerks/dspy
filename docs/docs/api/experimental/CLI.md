# dspy.experimental.CLI

!!! warning "Experimental API"
    `CLI` is experimental and may change without warning. It runs on POSIX systems only.

`CLI` is a predictor whose forward pass is a command: a coding agent run headless, or
any program that behaves like one. You give it a signature and an argv. Each call starts
the command, hands it the task, and returns a typed `Prediction`.

```python
import dspy
from dspy.experimental import CLI


class Locate(dspy.Signature):
    """Find where a question is answered in the documentation in the current directory."""

    question: str = dspy.InputField()
    files: list[str] = dspy.OutputField(desc="Relative paths of the files that state the answer.")
    quote: str = dspy.OutputField(desc="The sentence that answers it, copied exactly.")


locate = CLI(
    Locate,
    ["claude", "-p", "--allowedTools", "Read,Grep,Glob,Bash(submit:*)"],
    cwd="docs",
)

result = locate(question="How do I turn off caching?")
result.files  # a list[str]
```

No language model is configured. The agent brings its own model, tools and login; DSPy
contributes the signature, the demos, and the check on what comes back.

`CLI` is a `dspy.Predict` subclass and not a `dspy.LM`, because an agent takes a task
and returns a result, which is what a module does. For a command that answers one
prompt with one completion, see [Custom LM engines](../../tutorials/custom_lm_engines/index.md).

## How a call works

1. The signature's instructions, the demos, and the inputs are written to the command's
   standard input, followed by how to answer and the JSON Schema of the output fields.
2. `CLI` puts a `submit` program on the command's `PATH`. The agent finishes by running
   `submit '{"files": [...], "quote": "..."}'` from its shell.
3. The object is validated against the output fields' types with Pydantic. A submission
   that does not fit is answered with the reason and a non-zero exit status, so the
   agent can correct it and submit again in the same run.
4. The first accepted submission becomes the `Prediction`. The command has ten seconds
   to end its turn, and is then stopped along with its process group.

Nothing is parsed out of prose: there is no adapter and no fallback. A command that
ends without an accepted submission raises `dspy.experimental.CLIError`, which carries
`transcript`, `stderr`, `returncode`, and `rejected` (why each submission was refused).
This is the `SUBMIT` of [`dspy.RLM`](../modules/RLM.md), carried over a shell command
so that it needs nothing from an agent except the ability to run one.

The `Prediction` also has `transcript`, the end of what the command printed to
standard output, so no output field can have that name.

## Commands

The command is an argv, and `CLI` knows nothing about any particular agent. Model,
permissions, sandbox and session handling are expressed with the agent's own flags.
The agent must read its task from standard input and be allowed to run `submit`, which
writes into a temporary directory made for the call (under `TMPDIR`).

These were run end to end on Linux on 2026-10-01:

| Agent | Command | Result |
| --- | --- | --- |
| Claude Code 2.1.286 | `["claude", "-p", "--model", "haiku", "--no-session-persistence", "--allowedTools", "Read,Grep,Glob,Bash(submit:*)"]` | works |
| Codex CLI 0.158.0 | `["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "workspace-write", "-"]` | works |
| Codex CLI 0.158.0 | the same with `"--sandbox", "read-only"` | no submission: `submit` has to write a file |

Other agents have not been tried. One that takes its task as an argument can be
wrapped in a shell, which is still an argv: `["sh", "-c", 'exec my-agent "$(cat)"']`.

A command that prints its result instead of running `submit` can be piped into it,
because `submit` reads the object from standard input when given no argument:
`["sh", "-c", "my-tool --json | submit"]`. The output is validated like any other
submission. Such a command still receives the task on standard input, as text written
for an agent.

## With the rest of DSPy

`CLI` is found by `named_predictors()`, records its calls in the trace, and renders its
instructions and demos into the task. `dspy.Evaluate` and `dspy.BootstrapFewShot` have
been run on it unchanged against real agents, and `save` and `load` keep the
instructions and demos. `acall` runs the command without blocking the event loop, and
cancelling the task stops the command.

What assumes a language model behind the predictor has nothing to act on: `dspy.BestOfN`,
`dspy.Refine` and the later rounds of `BootstrapFewShot(max_rounds=...)` vary an LM's
rollout id. Optimizers that propose instructions have not been tried.

## Limits

- **It is not a sandbox.** The command runs with the authority of the current process,
  and the inputs reach an agent that may act on them. Grant it only what the task needs.
- **Calls are not cached or retried.** A call can change the working directory, so
  repeating one repeats the run, and a failed one raises.
- **`dspy.Evaluate` can start a slow call twice.** It resubmits a call that has been
  running for two minutes when three or fewer are left.
- **Interrupting a threaded run does not stop agents already started.** They run until
  they finish or time out. A call made from the main thread is stopped.
- **Stopping reaches the command's process group.** The command is sent `SIGTERM`
  first so that it can stop what it placed elsewhere, then the group is killed.
- **`save(..., save_program=True)` pickles the whole object**, command and `env` included.
- **No usage, cost, DSPy tools, or structured trajectory.** `transcript` is text.

A library that speaks each agent's own protocol can offer what this leaves out. `CLI`
is the opposite trade: nothing per agent, so nothing that depends on one.

::: dspy.experimental.CLI
    options:
        members: [__init__]
