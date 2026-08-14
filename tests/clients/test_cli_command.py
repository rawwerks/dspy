import pytest

from dspy.clients.cli_command import CLICommand, resolve_agent_cli


@pytest.mark.parametrize(
    ("name", "argv"),
    [
        ("claude", ("claude", "-p", "--output-format", "text", "--no-session-persistence")),
        ("codex", ("codex", "exec", "--ephemeral", "-")),
        ("gemini", ("gemini",)),
        ("pi", ("pi", "-p", "--no-session")),
        ("qwen", ("qwen",)),
    ],
)
def test_resolve_agent_cli_uses_documented_stdin_text_presets(name, argv):
    command = resolve_agent_cli(name)

    assert command == CLICommand(argv=argv)
    assert command.prompt_transport == "stdin"
    assert command.output_format == "text"


def test_resolve_agent_cli_appends_model_extra_args_and_codex_stdin_marker():
    command = resolve_agent_cli(
        "codex",
        model="gpt-5.6-luna",
        executable="/opt/codex",
        extra_args=("--sandbox", "read-only"),
    )

    assert command.argv == (
        "/opt/codex",
        "exec",
        "--ephemeral",
        "--model",
        "gpt-5.6-luna",
        "--sandbox",
        "read-only",
        "-",
    )


def test_resolve_agent_cli_inserts_prefix_args_before_codex_subcommand():
    command = resolve_agent_cli(
        "codex",
        prefix_args=("-a", "never"),
        extra_args=("--sandbox", "read-only"),
    )

    assert command.argv == (
        "codex",
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "-",
    )


def test_resolve_agent_cli_rejects_unknown_or_ambiguous_inputs():
    with pytest.raises(ValueError, match="Available presets"):
        resolve_agent_cli("cursor")
    with pytest.raises(TypeError, match="extra_args"):
        resolve_agent_cli("codex", extra_args="--json")
    with pytest.raises(TypeError, match="prefix_args"):
        resolve_agent_cli("codex", prefix_args="-a never")
    with pytest.raises(TypeError, match="prefix_args"):
        resolve_agent_cli("codex", prefix_args=("",))
