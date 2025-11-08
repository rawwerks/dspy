"""CLILM example that uses ``codex exec --json`` to stream JSON events.

The JSON mode of Codex CLI already emits JSON Lines with ``item.completed``
events, which means the :class:`dspy.clients.cli_lm.CLILM` helper can extract
the final assistant message without any extra parsing on our end. Just like the
other scripts in this directory, this file acts as both a runnable demo and a
bridge that can be used directly as the CLILM command.
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


def _codex_bin() -> str:
    return os.environ.get("CODEX_BIN", "codex")


def _run_bridge() -> int:
    prompt = sys.stdin.read()
    if not prompt.strip():
        print("[codex-json-bridge] Refusing to run with an empty prompt.", file=sys.stderr)
        return 2

    command = [_codex_bin(), "exec", "--json", prompt]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print(
            f"[codex-json-bridge] Unable to find '{command[0]}'. "
            "Set CODEX_BIN to point to the Codex CLI.",
            file=sys.stderr,
        )
        return 127

    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_example(question: str) -> None:
    bridge_command = [sys.executable, str(Path(__file__).resolve()), "--bridge"]
    lm = CLILM(bridge_command)
    dspy.configure(lm=lm)

    predictor = dspy.Predict(SimpleMath)
    result = predictor(question=question)

    print(f"Question: {question}")
    print(f"Codex answer (json mode): {result.answer}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="If set, run as a CLI bridge that forwards stdin to codex exec --json.",
    )
    parser.add_argument(
        "--question",
        default="List three US presidents.",
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
