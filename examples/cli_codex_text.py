"""CLILM example that routes DSPy prompts to ``codex exec`` in text mode.

Like the Claude examples, this file doubles as both a demo and a bridge:

* Without arguments it configures DSPy to talk to Codex CLI via CLILM and runs
  a tiny ``SimpleMath`` predictor.
* With ``--bridge`` it reads the prompt from ``stdin`` (as CLILM provides it)
  and forwards the text to ``codex exec" so that the Codex CLI can generate a
  response.

Environment variables:

``CODEX_BIN`` (optional)
    Path to the Codex CLI binary. Defaults to ``"codex"``.
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
        print("[codex-text-bridge] Refusing to run with an empty prompt.", file=sys.stderr)
        return 2

    command = [_codex_bin(), "exec", prompt]

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print(
            f"[codex-text-bridge] Unable to find '{command[0]}'. "
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
    print(f"Codex answer: {result.answer}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="If set, run as a CLI bridge that forwards stdin to codex exec.",
    )
    parser.add_argument(
        "--question",
        default="Explain the Pythagorean theorem in one sentence.",
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
