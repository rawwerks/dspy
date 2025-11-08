"""Deterministic CLI process used by CLILM tests."""

from __future__ import annotations

import json
import os
import re
import sys


FIELD_PATTERN = re.compile(r"\[\[\s*##\s*(?P<field>\w+)\s*##\s*\]\]\s*(?P<value>.*?)\s*(?=\[\[|$)", re.DOTALL)


def _extract_field(text: str, field: str) -> str:
    last_value = None
    for match in FIELD_PATTERN.finditer(text):
        if match.group("field").lower() == field.lower():
            candidate = match.group("value").strip()
            if "\n\n" in candidate:
                candidate = candidate.split("\n\n", 1)[0].strip()
            last_value = candidate
    return last_value if last_value is not None else text.strip()


def main() -> int:
    prompt = sys.stdin.read()
    if not prompt:
        print("no prompt provided", file=sys.stderr)
        return 2

    mode = os.environ.get("CLI_MODE")

    if mode == "fail":
        print("intentional failure", file=sys.stderr)
        return 2

    if mode == "invalid":
        sys.stdout.write("{not json")
        return 0

    question = _extract_field(prompt, "question")
    env_hint = os.environ.get("CLI_ADAPTER_ECHO_ENV")
    index = int(os.environ.get("CLI_GENERATION_INDEX", "0")) + 1

    suffix = f" [{env_hint}]" if env_hint else ""
    answer_text = f"Echo {index}: {question}{suffix}"

    if mode == "json":
        events = [
            {"type": "thread.started"},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": f"[[ ## answer ## ]]\n{answer_text}\n\n[[ ## completed ## ]]"},
            },
        ]
        for event in events:
            sys.stdout.write(json.dumps(event) + "\n")
    else:
        result = (
            "[[ ## answer ## ]]\n"
            f"{answer_text}\n\n"
            "[[ ## completed ## ]]"
        )
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
