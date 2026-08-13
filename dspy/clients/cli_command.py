"""Optional command presets for one-shot agent CLI calls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

AgentCLIName = Literal["claude", "codex", "gemini", "pi", "qwen"]
PromptTransport = Literal["stdin"]
OutputFormat = Literal["text"]


@dataclass(frozen=True)
class CLICommand:
    """A provider-neutral one-shot CLI invocation.

    Presets deliberately cover only text over stdin/stdout. Permissions,
    sandboxing, sessions, and other agent-specific policy remain explicit
    command arguments supplied by the caller.
    """

    argv: tuple[str, ...]
    prompt_transport: PromptTransport = "stdin"
    output_format: OutputFormat = "text"


@dataclass(frozen=True)
class _Preset:
    argv: tuple[str, ...]
    suffix: tuple[str, ...] = ()
    model_flag: str | None = "--model"


_PRESETS = {
    "claude": _Preset(("claude", "-p", "--output-format", "text", "--no-session-persistence")),
    "codex": _Preset(("codex", "exec"), suffix=("-",)),
    "gemini": _Preset(("gemini",)),
    "pi": _Preset(("pi", "-p", "--no-session")),
    "qwen": _Preset(("qwen",)),
}


def resolve_agent_cli(
    name: AgentCLIName,
    *,
    model: str | None = None,
    executable: str | None = None,
    prefix_args: Sequence[str] = (),
    extra_args: Sequence[str] = (),
) -> CLICommand:
    """Resolve a documented one-shot agent CLI preset.

    This is a convenience function, not a compatibility guarantee. Agent CLIs
    change independently, so callers can override the executable, append
    arguments, or pass explicit argv directly to :class:`dspy.CLI`.
    """
    try:
        preset = _PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown agent CLI {name!r}. Available presets: {available}") from exc

    if executable is not None and not executable:
        raise ValueError("executable cannot be empty")
    for argument_name, arguments in (("prefix_args", prefix_args), ("extra_args", extra_args)):
        if isinstance(arguments, (str, bytes)) or any(not isinstance(arg, str) or not arg for arg in arguments):
            raise TypeError(f"{argument_name} must be a sequence of non-empty strings")

    argv = list(preset.argv)
    if executable is not None:
        argv[0] = executable
    argv[1:1] = prefix_args
    if model is not None:
        if not model:
            raise ValueError("model cannot be empty")
        if preset.model_flag is None:
            raise ValueError(f"The {name!r} preset does not define a model flag")
        argv.extend((preset.model_flag, model))
    argv.extend(extra_args)
    argv.extend(preset.suffix)
    return CLICommand(argv=tuple(argv))


__all__ = ["AgentCLIName", "CLICommand", "OutputFormat", "PromptTransport", "resolve_agent_cli"]
