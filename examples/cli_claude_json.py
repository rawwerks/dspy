"""CLILM example that proxies through ``claude -p`` in JSON output mode.

This script mirrors :mod:`examples.cli_claude_text`, but when run in bridge
mode it calls ``claude`` with ``--output-format json`` and converts the response
into the event stream format that :class:`dspy.clients.cli_lm.CLILM` looks for.

Usage::

    # Run the DSPy demo (requires Claude CLI authentication)
    python examples/cli_claude_json.py

    # Use the file purely as a bridge for CLILM
    python examples/cli_claude_json.py --bridge

Environment variables:

``CLAUDE_BIN`` (optional)
    Path to the Claude CLI binary. Defaults to ``"claude"``.
"""

from __future__ import annotations

import argparse
import json
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


def _as_event_stream(text: str) -> str:
    """Format ``text`` as the JSONL ``item.completed`` event CLILM expects."""

    event = {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
    return json.dumps(event, ensure_ascii=False) + "\n"


def _run_bridge() -> int:
    prompt = sys.stdin.read()
    if not prompt.strip():
        print("[claude-json-bridge] Refusing to run with an empty prompt.", file=sys.stderr)
        return 2

    command = [_claude_bin(), "-p", prompt, "--output-format", "json"]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print(
            f"[claude-json-bridge] Unable to find '{command[0]}'. "
            "Set CLAUDE_BIN to point to the Claude CLI.",
            file=sys.stderr,
        )
        return 127

    stdout = proc.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            sys.stdout.write(stdout + "\n")
        else:
            completion = payload.get("result") or payload.get("output") or payload.get("text") or ""
            sys.stdout.write(_as_event_stream(completion or stdout))
    sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_example(question: str) -> None:
    bridge_command = [sys.executable, str(Path(__file__).resolve()), "--bridge"]
    lm = CLILM(bridge_command)
    dspy.configure(lm=lm)

    predictor = dspy.Predict(SimpleMath)
    result = predictor(question=question)

    print(f"Question: {question}")
    print(f"Claude answer (json mode): {result.answer}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="If set, run as a CLI bridge that forwards stdin to the Claude CLI.",
    )
    parser.add_argument(
        "--question",
        default="What is the capital of France?",
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
