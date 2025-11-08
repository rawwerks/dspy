import json
import os
import sys

MODE = os.environ.get("CLI_MODE", "plain")


def _extract_user_message(text: str) -> str:
    parts = [segment.strip() for segment in text.split("USER:") if segment.strip()]
    if not parts:
        return text.strip()
    return parts[-1]


def main():
    payload = sys.stdin.read()
    if not payload:
        print("failed to read input", file=sys.stderr)
        return 3

    content = _extract_user_message(payload)

    if MODE == "fail":
        print("intentional failure", file=sys.stderr)
        return 2

    if MODE == "invalid":
        sys.stdout.write("{not json")
        return 0

    if MODE == "warn":
        print("cli warning: proceed with caution", file=sys.stderr)
        sys.stdout.write(content)
        return 0

    if MODE == "json":
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": content}},
        ]
        for event in events:
            sys.stdout.write(json.dumps(event) + "\n")
        return 0

    sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
