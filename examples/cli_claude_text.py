"""End-to-end example of using CLILM with the Claude CLI in text mode.

This module serves two purposes:

1. When executed normally (no additional flags) it configures DSPy to use
   :class:`dspy.clients.cli_lm.CLILM` backed by the Claude CLI and runs a tiny
   ``SimpleMath`` predictor so you can see the response end-to-end.
2. When executed with ``--bridge`` it acts as a thin wrapper that CLILM can use
   as the CLI command. The wrapper reads the prompt from ``stdin`` and forwards
   it to ``claude -p`` so that the DSPy conversation is preserved.

Usage::

    # Run the demo (requires the Claude CLI to be authenticated locally)
    python examples/cli_claude_text.py

    # Use the file purely as a bridge for CLILM
    python examples/cli_claude_text.py --bridge

Environment variables:

``CLAUDE_BIN`` (optional)
    Path to the Claude CLI binary. Defaults to ``"claude"``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import dspy
from dspy.clients.cli_lm import CLILM


class SimpleMath(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _claude_bin() -> str:
    return os.environ.get("CLAUDE_BIN", "claude")


def _run_bridge() -> int:
    """Forward the DSPy prompt to ``claude -p`` and relay its output."""

    prompt = sys.stdin.read()
    if not prompt.strip():
        print("[claude-text-bridge] Refusing to run with an empty prompt.", file=sys.stderr)
        return 2

    command = [_claude_bin(), "-p", prompt]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print(
            f"[claude-text-bridge] Unable to find '{command[0]}'. "
            "Set CLAUDE_BIN to point to the Claude CLI.",
            file=sys.stderr,
        )
        return 127

    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_example(question: str) -> None:
    """Configure DSPy with CLILM+Claude and print the model's response."""

    bridge_command = [sys.executable, str(Path(__file__).resolve()), "--bridge"]
    lm = CLILM(bridge_command)
    dspy.configure(lm=lm)

    predictor = dspy.Predict(SimpleMath)
    result = predictor(question=question)

    print(f"Question: {question}")
    print(f"Claude answer: {result.answer}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="If set, run as a CLI bridge that forwards stdin to the Claude CLI.",
    )
    parser.add_argument(
        "--question",
        default="What is 2 + 2?",
        help="Question to send when running the demo mode (ignored for --bridge).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bridge:
        raise SystemExit(_run_bridge())
    _run_example(question=args.question)


if __name__ == "__main__":
    main()
