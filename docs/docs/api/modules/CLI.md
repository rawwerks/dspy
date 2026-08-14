# dspy.CLI

`CLI` runs one ordinary DSPy `Predict` through an arbitrary text command. DSPy's
adapter renders the signature and inputs, `CLI` writes that prompt to standard
input, and the same adapter parses standard output into the signature's typed
fields.

```python
import sys

import dspy

program = dspy.CLI(
    "question -> answer",
    command=[sys.executable, "my_agent.py"],
    timeout=30,
)
result = program(question="What is the capital of France?")
print(result.answer)
```

`command` is an argv sequence, not a shell command string. Each generation
starts one subprocess (`config={"n": 3}` starts three independent processes),
which receives `DSPY_CLI_GENERATION_INDEX` and `DSPY_CLI_TOTAL_GENERATIONS`
environment variables; caller-supplied `env=` values override them.
The command must read the complete prompt from stdin, write only the
DSPy-formatted response to stdout, and signal failure with a nonzero exit
status. Diagnostic output belongs on stderr.

The module exposes exactly one optimizer-visible predictor named `predict`, so
standard optimizers tune the original signature rather than a separate wrapper
protocol. It does not use or modify the globally configured LM, and saved
program state contains the tuned prompts and demos but never the command —
programs are reconstructed from `__init__`, like other execution-backed
modules.

Unless an adapter is configured globally, calls use a `ChatAdapter` with the
automatic JSON-fallback retry disabled, so a successful process whose stdout
cannot be parsed fails after exactly one command invocation. A globally
configured adapter is honored as-is; note that an adapter that retries on
parse failure will re-invoke the command. Calls default to a 600-second
`timeout` because an agent blocked waiting for interaction the transport cannot
provide would otherwise hang forever; pass `timeout=None` only when unbounded
execution is intended.

For optional agent CLI presets, use `dspy.resolve_agent_cli`:

```python
command = dspy.resolve_agent_cli(
    "codex",
    prefix_args=["-a", "never"],
    extra_args=["--sandbox", "read-only"],
)
program = dspy.CLI("task -> result", command=command)
```

Presets add only documented headless/stdin/text arguments. Permissions,
sandboxing, sessions, and other agent-specific policy remain explicit caller
arguments.

## What `CLI` deliberately does not manage

`CLI` owns the transport contract — prompt in over stdin, typed fields parsed
from stdout, bounded time and output — and nothing else. Each concern below is
excluded on purpose, because its semantics differ across agents and change as
agent CLIs evolve. A uniform DSPy option would be a promise this module cannot
keep. Every one of them can be expressed today as explicit command arguments.

**Sandboxing.** Agents differ in what "sandbox" means (filesystem scope,
network, shell execution), so `CLI` neither adds nor removes one. Pass the
agent's own flags:

```python
command = dspy.resolve_agent_cli("codex", extra_args=["--sandbox", "read-only"])
```

**Approvals and permissions.** Granting an agent authority to act — up to
unattended "yolo" execution — is a decision only the caller can own. Because
the prompt is written to stdin and stdin is then closed, an agent can never
ask for approval mid-call; authority must be declared up front, and headless
agents deny what was not granted rather than prompting. Three degrees:

```python
# Degree 0 — no flags: pure generation plus whatever the agent allows by
# default in headless mode (typically reads). Sufficient for extraction,
# classification, review, and Q&A signatures.
program = dspy.CLI("question -> answer", dspy.resolve_agent_cli("claude"))

# Degree 1 — scoped autonomy: auto-approved work inside a declared boundary.
command = dspy.resolve_agent_cli("codex", extra_args=["--sandbox", "workspace-write"])
command = dspy.resolve_agent_cli("claude", extra_args=["--allowedTools", "Read Grep Bash(git diff:*)"])
command = dspy.resolve_agent_cli("gemini", extra_args=["--approval-mode", "auto_edit"])

# Degree 2 — full bypass, for externally sandboxed environments only.
command = dspy.resolve_agent_cli("claude", extra_args=["--dangerously-skip-permissions"])
command = dspy.resolve_agent_cli("codex", extra_args=["--dangerously-bypass-approvals-and-sandbox"])
```

Agents also apply their own host-level configuration underneath these flags —
for example Claude Code merges the user's persistent permission allowlist
(`~/.claude/settings.json`) into headless runs. The same program can therefore
hold different authority on different machines; declare policy explicitly when
that matters.

**Sessions and persisted state.** DSPy programs assume each call is
independent, so presets pin statelessness where the agent documents a flag for
it (`claude --no-session-persistence`, `codex exec --ephemeral`,
`pi --no-session`). The `gemini` and `qwen` presets have no documented
equivalent; those agents may persist per-call session state, which matters
under `dspy.Evaluate` or optimizers that make many calls.

**Filesystem and environment scope.** `cwd=` sets where the agent runs and
`env=` adds variables on top of the inherited environment. Point `cwd` at the
directory you intend the agent to see; use the agent's own flags (for example
`claude --add-dir`) to widen access rather than running from a broader
directory. Note that agents apply their own workspace policy relative to `cwd`
— for example `codex exec` refuses to run outside a trusted or git-managed
directory unless passed `--skip-git-repo-check`.

Flags above are current as of their CLIs' 2026-08 releases; agent CLIs change
independently of DSPy, so verify against `--help` when a command misbehaves.

`CLI` runs the requested executable directly on the host; it does not provide a
sandbox. The child process inherits the full parent environment — including
any secrets in it — with `env=` merged on top; server-side deployments should
scrub the environment with a wrapper script when running tool-capable agents
on untrusted inputs. Native tool calls and non-text LM content are rejected
because plain stdin/stdout defines no provider-neutral representation for
them. stdout and stderr are bounded independently by `max_output_bytes`, and timeout,
cancellation, or output overflow terminates the subprocess process group.

<!-- START_API_REF -->
::: dspy.CLI
    handler: python
    options:
        members:
            - __call__
            - acall
            - forward
            - aforward
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
