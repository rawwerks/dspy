"""Simple CLI agent for validating the CLIAdapter workflow.

The agent reads a JSON payload from stdin (matching the structure emitted by
CLIAdapter), and writes a JSON object containing the generated completions to
stdout. It supports a few modes that are handy for tests:

- default / ``echo``: mirrors the question back in the adapter's output format
- ``multi``: identical to echo but kept for backwards compatibility in tests
- ``fail``: writes an error to stderr and exits with a non-zero status
- ``invalid_json``: writes malformed JSON to stdout
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _ensure(condition: bool, message: str) -> None:
    if condition:
        return
    print(message, file=sys.stderr)
    sys.exit(3)


def _load_payload() -> dict:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        print(f"failed to decode payload: {exc}", file=sys.stderr)
        sys.exit(3)


def _format_completion(text: str) -> dict[str, str]:
    completion = f"[[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]"
    return {"text": completion}


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal CLI agent for CLIAdapter demos")
    parser.add_argument(
        "mode",
        choices=["echo", "multi", "fail", "invalid_json"],
        nargs="?",
        default="echo",
        help="Select the agent behavior",
    )
    args = parser.parse_args()

    payload = _load_payload()
    _ensure("messages" in payload, "payload missing messages")
    _ensure("inputs" in payload, "payload missing inputs")
    _ensure("signature" in payload, "payload missing signature metadata")

    if args.mode == "fail":
        print("intentional failure", file=sys.stderr)
        return 2

    if args.mode == "invalid_json":
        sys.stdout.write("{not json")
        return 0

    question = payload["inputs"].get("question", "")
    n = int(payload.get("lm_kwargs", {}).get("n", 1))
    env_hint = os.environ.get("CLI_ADAPTER_ECHO_ENV")

    outputs = []
    for idx in range(n):
        suffix = f" [{env_hint}]" if env_hint else ""
        text = f"Echo {idx + 1}: {question}{suffix}".strip()
        outputs.append(_format_completion(text))

    json.dump({"outputs": outputs}, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
