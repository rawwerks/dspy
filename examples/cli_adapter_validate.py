from __future__ import annotations

import sys
from pathlib import Path

import dspy
from dspy.adapters.cli_adapter import CLIAdapter
from dspy.utils.dummies import DummyLM


class SimpleMath(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def main() -> None:
    agent_path = Path(__file__).resolve().parents[1] / "agent.py"
    adapter = CLIAdapter([sys.executable, str(agent_path)])

    dummy_lm = DummyLM([{"answer": ""}])
    dspy.configure(lm=dummy_lm, adapter=adapter)

    predictor = dspy.Predict(SimpleMath)
    result = predictor(question="What is 2 + 2?")

    expected = "Echo 1: What is 2 + 2?"
    if result.answer != expected:
        raise SystemExit(f"Unexpected adapter output: {result.answer}")

    print(f"CLIAdapter produced: {result.answer}")


if __name__ == "__main__":
    main()
