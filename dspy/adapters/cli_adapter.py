"""Adapter tailored for CLI-driven language models."""

from __future__ import annotations

from dspy.adapters.chat_adapter import ChatAdapter


class CLIAdapter(ChatAdapter):
    """A thin wrapper around :class:`ChatAdapter` for CLI workflows.

    This adapter keeps the familiar DSPy prompt/parse structure but is commonly
    paired with :class:`dspy.clients.cli_lm.CLILM`, which executes an external
    CLI command (e.g., ``codex exec``) using the formatted prompt as stdin.
    """

    pass
