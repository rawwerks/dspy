from __future__ import annotations

import dspy
from dspy.adapters.cli_adapter import CLIAdapter
from dspy.clients.cli_lm import CLILM


class SimpleMath(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def main() -> None:
    command = [
        "codex",
        "--ask-for-approval",
        "never",
        "--sandbox",
        "workspace-write",
        "exec",
        "--json",
    ]
    adapter = CLIAdapter()
    lm = CLILM(command)

    dspy.configure(lm=lm, adapter=adapter)

    predictor = dspy.Predict(SimpleMath)
    result = predictor(question="What is 2 + 2?")

    print(f"CLIAdapter (via Codex CLI) answered: {result.answer}")


if __name__ == "__main__":
    main()
