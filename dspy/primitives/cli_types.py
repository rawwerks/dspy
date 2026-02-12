"""
Data types for CLI module execution tracing and sandboxing.

CLIEvent and CLITrajectory capture the full record of a CLI subprocess
execution, analogous to REPLEntry/REPLHistory for RLM.

CLISandbox defines the protocol for subprocess sandboxing, analogous to
RLM's CodeInterpreter protocol.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel


class CLIEvent(BaseModel):
    """A single event from a CLI agent's structured output stream.

    Many CLI agents (Codex, Claude Code) emit structured JSONL events.
    This captures individual events for optimizer inspection.
    """

    type: str
    """Event type: 'thinking', 'tool_call', 'tool_result', 'agent_message', 'error', etc."""

    content: str
    """The event payload text."""

    timestamp: float | None = None
    """Optional timestamp from the event."""

    raw: dict[str, Any] | None = None
    """The original parsed JSON event, if available."""

    def format(self) -> str:
        """Human-readable format for inclusion in Predict inputs."""
        ts = f" (t={self.timestamp:.2f}s)" if self.timestamp is not None else ""
        return f"[{self.type}]{ts}: {self.content}"


class CLITrajectory(BaseModel):
    """Full record of a CLI execution for optimizer inspection.

    Analogous to REPLHistory for RLM. Captures everything needed for
    optimizers (especially GEPA) to understand what the CLI agent did.
    """

    prompt: str
    """The prompt text sent to the CLI."""

    events: list[CLIEvent]
    """Parsed structured events (empty for plain-text CLIs)."""

    raw_stdout: str
    """Full stdout from the subprocess."""

    stderr: str
    """Full stderr from the subprocess."""

    returncode: int
    """Process exit code."""

    elapsed: float
    """Wall-clock seconds for the subprocess execution."""

    def format(self, max_chars: int = 10_000) -> str:
        """Human-readable format for inclusion in Predict inputs.

        Args:
            max_chars: Maximum characters for stdout/stderr sections.
        """
        parts = [f"=== CLI Execution (exit={self.returncode}, {self.elapsed:.1f}s) ==="]
        parts.append(f"\n--- Prompt ---\n{self.prompt}")

        if self.events:
            parts.append(f"\n--- Events ({len(self.events)}) ---")
            for event in self.events:
                parts.append(event.format())

        stdout_display = self.raw_stdout
        if len(stdout_display) > max_chars:
            stdout_display = stdout_display[:max_chars] + "\n... (truncated)"
        parts.append(f"\n--- Stdout ---\n{stdout_display}")

        if self.stderr.strip():
            stderr_display = self.stderr
            if len(stderr_display) > max_chars:
                stderr_display = stderr_display[:max_chars] + "\n... (truncated)"
            parts.append(f"\n--- Stderr ---\n{stderr_display}")

        return "\n".join(parts)

    def __str__(self) -> str:
        return self.format()

    def get_agent_message(self) -> str | None:
        """Extract the final agent message from events, if any.

        Looks for the last 'agent_message' type event, which is the
        pattern used by Codex and Claude Code JSONL output.
        """
        for event in reversed(self.events):
            if event.type == "agent_message" and event.content.strip():
                return event.content.strip()
        return None


def parse_jsonl_events(stdout: str) -> list[CLIEvent]:
    """Parse JSONL-formatted stdout into CLIEvent objects.

    Handles mixed output (some lines JSON, some not) gracefully.
    Recognizes patterns from Codex and Claude Code output formats.

    Args:
        stdout: Raw stdout text, potentially containing JSONL lines.

    Returns:
        List of parsed CLIEvent objects. Empty if no valid JSONL found.
    """
    events: list[CLIEvent] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        event = _parse_single_event(data)
        if event is not None:
            events.append(event)

    return events


def _parse_single_event(data: dict[str, Any]) -> CLIEvent | None:
    """Parse a single JSON object into a CLIEvent.

    Handles multiple formats:
    - Codex-style: {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
    - Generic: {"type": "...", "content": "..."} or {"type": "...", "text": "..."}
    """
    event_type = data.get("type")
    if not event_type:
        return None

    # Codex-style nested events
    if event_type == "item.completed":
        item = data.get("item")
        if isinstance(item, dict):
            item_type = item.get("type", "unknown")
            text = item.get("text") or item.get("content") or ""
            return CLIEvent(
                type=item_type,
                content=str(text),
                raw=data,
            )

    # Generic event format
    content = data.get("content") or data.get("text") or data.get("message") or ""
    timestamp = data.get("timestamp") or data.get("time")

    return CLIEvent(
        type=event_type,
        content=str(content),
        timestamp=float(timestamp) if timestamp is not None else None,
        raw=data,
    )


# =============================================================================
# Sandbox Protocol
# =============================================================================


@runtime_checkable
class CLISandbox(Protocol):
    """Protocol for CLI subprocess sandboxing.

    Analogous to RLM's CodeInterpreter protocol. Implementations wrap
    the CLI command to run it in a restricted environment.

    Built-in implementations:
    - DenoSandbox: Uses Deno's permission system (same as PythonInterpreter)
    - BubbleSandbox: Uses bubblewrap (bwrap) for Linux namespace isolation
    - DockerSandbox: Runs the CLI inside a Docker container

    Custom implementations can use any isolation mechanism (firejail,
    nsjail, gVisor, etc.) — just implement wrap_command().

    Example:
        ```python
        class MySandbox:
            def wrap_command(self, command, cwd=None, env=None):
                return ["sandbox-tool", "--restrict", "--"] + list(command)

        cli = dspy.CLI("task -> result", command="my-cli", sandbox=MySandbox())
        ```
    """

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Wrap a CLI command to run inside the sandbox.

        Args:
            command: The original command to execute.
            cwd: Working directory the CLI expects to run in.
            env: Environment variables the CLI needs.

        Returns:
            The wrapped command list that runs inside the sandbox.
        """
        ...


class DenoSandbox:
    """Sandbox using Deno's permission system.

    Uses the same permission model as ``PythonInterpreter`` from RLM:
    Deno's V8 security sandbox restricts filesystem, network, environment,
    and subprocess access by default — you explicitly allow what's needed.

    The CLI command is spawned as a subprocess inside Deno via
    ``Deno.Command``, inheriting only the permissions you grant.

    Requires ``deno`` to be installed (already a dependency if using RLM).

    Args:
        enable_run: Programs the CLI is allowed to spawn. The first element
            of your CLI command is added automatically.
        enable_read_paths: Files/directories to allow reading from.
        enable_write_paths: Files/directories to allow writing to.
        enable_env_vars: Environment variable names to allow.
        enable_network_access: Domains/IPs to allow network access to.

    Example:
        ```python
        sandbox = DenoSandbox(
            enable_read_paths=["/data"],
            enable_write_paths=["/output"],
            enable_network_access=["api.openai.com"],
        )
        cli = dspy.CLI("task -> result", command="my-cli", sandbox=sandbox)
        ```
    """

    def __init__(
        self,
        *,
        enable_run: Sequence[str] | None = None,
        enable_read_paths: Sequence[str] | None = None,
        enable_write_paths: Sequence[str] | None = None,
        enable_env_vars: Sequence[str] | None = None,
        enable_network_access: Sequence[str] | None = None,
    ):
        self.enable_run = list(enable_run or [])
        self.enable_read_paths = list(enable_read_paths or [])
        self.enable_write_paths = list(enable_write_paths or [])
        self.enable_env_vars = list(enable_env_vars or [])
        self.enable_network_access = list(enable_network_access or [])

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        import json as _json
        import shutil

        command = list(command)

        # Find deno
        deno = shutil.which("deno")
        if not deno:
            raise RuntimeError(
                "DenoSandbox requires 'deno' to be installed. "
                "See https://docs.deno.com/runtime/getting_started/installation/"
            )

        # Build deno permission flags
        args = ["deno", "run"]

        # --allow-run: always include the CLI binary itself
        run_list = list(self.enable_run)
        if command and command[0] not in run_list:
            run_list.append(command[0])
        args.append(f"--allow-run={','.join(run_list)}")

        # --allow-read
        read_paths = list(self.enable_read_paths)
        if cwd and cwd not in read_paths:
            read_paths.append(cwd)
        if read_paths:
            args.append(f"--allow-read={','.join(read_paths)}")

        # --allow-write
        write_paths = list(self.enable_write_paths)
        if cwd and cwd not in write_paths:
            write_paths.append(cwd)
        if write_paths:
            args.append(f"--allow-write={','.join(write_paths)}")

        # --allow-env
        env_vars = list(self.enable_env_vars)
        if env:
            for key in env:
                if key not in env_vars:
                    env_vars.append(key)
        if env_vars:
            args.append(f"--allow-env={','.join(env_vars)}")

        # --allow-net
        if self.enable_network_access:
            args.append(f"--allow-net={','.join(self.enable_network_access)}")

        # Inline JS runner that spawns the command and inherits stdio.
        # Uses `deno run [flags] --` with inline eval via process substitution
        # would be shell-dependent, so we write a temp file instead.
        import tempfile

        cmd_json = _json.dumps(command[0])
        args_json = _json.dumps(command[1:])
        cwd_json = _json.dumps(cwd) if cwd else "undefined"

        js = (
            f"const cmd = new Deno.Command({cmd_json}, {{"
            f"args: {args_json}, "
            f"cwd: {cwd_json}, "
            f"stdin: 'inherit', stdout: 'inherit', stderr: 'inherit'"
            f"}}); "
            f"const status = cmd.outputSync(); "
            f"Deno.exit(status.code);"
        )

        # Write runner to a temp .ts file (auto-cleaned by OS)
        runner = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", prefix="dspy_deno_sandbox_", delete=False,
        )
        runner.write(js)
        runner.close()

        # Allow deno to read the runner script
        read_flag_idx = next(
            (i for i, a in enumerate(args) if a.startswith("--allow-read=")), None
        )
        if read_flag_idx is not None:
            args[read_flag_idx] += f",{runner.name}"
        else:
            args.append(f"--allow-read={runner.name}")

        args.append(runner.name)
        return args


class BubbleSandbox:
    """Sandbox using bubblewrap (bwrap) for Linux namespace isolation.

    Provides filesystem and network isolation by default. The working
    directory is bind-mounted read-write; everything else is read-only
    or hidden.

    Requires ``bwrap`` to be installed (``pacman -S bubblewrap`` or
    ``apt install bubblewrap``).

    Args:
        allow_network: Whether to allow network access (default False).
        allow_paths: Additional paths to bind-mount read-write.
        read_only_paths: Additional paths to bind-mount read-only.
            Defaults to common system paths (/usr, /lib, /etc, /bin).
        share_home: Whether to share the user's home directory read-only
            (default False).
    """

    def __init__(
        self,
        *,
        allow_network: bool = False,
        allow_paths: Sequence[str] | None = None,
        read_only_paths: Sequence[str] | None = None,
        share_home: bool = False,
    ):
        self.allow_network = allow_network
        self.allow_paths = list(allow_paths or [])
        self.read_only_paths = list(read_only_paths or ["/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin"])
        self.share_home = share_home

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        import os

        cmd = ["bwrap"]

        # Basic isolation
        cmd.extend(["--unshare-pid", "--die-with-parent", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"])

        if not self.allow_network:
            cmd.append("--unshare-net")

        # Read-only system paths
        for path in self.read_only_paths:
            if os.path.exists(path):
                cmd.extend(["--ro-bind", path, path])

        # Home directory
        if self.share_home:
            home = os.path.expanduser("~")
            cmd.extend(["--ro-bind", home, home])

        # Working directory (read-write)
        if cwd:
            cmd.extend(["--bind", cwd, cwd, "--chdir", cwd])

        # Additional read-write paths
        for path in self.allow_paths:
            if os.path.exists(path):
                cmd.extend(["--bind", path, path])

        cmd.extend(["--"] + list(command))
        return cmd


class DockerSandbox:
    """Sandbox using Docker containers.

    Runs the CLI command inside a Docker container with configurable
    isolation.

    Args:
        image: Docker image to use (default "python:3.12-slim").
        allow_network: Whether to allow network access (default False).
        mount_cwd: Whether to mount the working directory (default True).
        extra_args: Additional ``docker run`` arguments.
    """

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        allow_network: bool = False,
        mount_cwd: bool = True,
        extra_args: Sequence[str] | None = None,
    ):
        self.image = image
        self.allow_network = allow_network
        self.mount_cwd = mount_cwd
        self.extra_args = list(extra_args or [])

    def wrap_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        cmd = ["docker", "run", "--rm", "-i"]

        if not self.allow_network:
            cmd.extend(["--network", "none"])

        if self.mount_cwd and cwd:
            cmd.extend(["-v", f"{cwd}:{cwd}", "-w", cwd])

        # Pass environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.extend(self.extra_args)
        cmd.append(self.image)
        cmd.extend(command)
        return cmd
