"""The python blocks of the dspy.experimental.RDM reference page, run top to bottom as one program.

Every `print` on the page is followed by the lines it prints, as comments, and the page is only
right if the program prints them. By default stand-ins answer for the decision model and the
generative model, and the page's REPL is the in-process interpreter of dspy-interpreters. With
`--llm_call`, Jev and a generative model through OpenRouter answer (TYPESAFE_API_KEY and
OPENROUTER_API_KEY), the REPL is the Deno sandbox, and the page must still be right.
"""

import contextlib
import io
import json
import os
import re
from pathlib import Path

import pytest

import dspy
from dspy.experimental import TypeSafe
from dspy.utils.dummies import DummyLM

PAGE = Path(__file__).parents[2] / "docs/docs/api/experimental/RDM.md"


def blocks():
    return re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S)


def documented(blocks):
    """What the page says its program prints: the comment lines right after each top-level `print`."""
    lines = []
    for block in blocks:
        rows = block.splitlines()
        for number, row in enumerate(rows):
            if row.startswith("print("):
                lines += [comment[2:] for comment in _comments(rows[number + 1 :])]
    return lines


def _comments(rows):
    for row in rows:
        if not row.startswith("# "):
            return
        yield row


class Jev:
    """A stand-in System One client, answering each question by a rule written for the page's examples."""

    supports_decision_requests = True
    cache = False
    model = "jev-stand-in"

    def __init__(self, model=None):
        self.history, self.callbacks = [], []
        self.plans = {
            "Lock the front door.": iter(["lock locked", "door closed", "lock locked", "finish"]),
            "Good night: everything off, and lock up.": iter(["tv off", "lights off", "lock locked", "door closed", "lock locked", "finish"]),
            "Night!": iter(["good night", "finish"]),
            "Bring the heat down to 18 degrees.": iter(["write", "cool_to_18", "finish"]),
        }

    def answer(self, inputs, name, question):
        shown = json.dumps({key: value for key, value in inputs.items() if key not in ("question", "questions")})
        if name == "next":
            return next(self.plans[inputs["task"]])
        if name == "category":
            return next(option for option in question["criteria"] if option in ("Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"))
        request = inputs.get("task", "")
        return {
            "screen": "tv" in shown.lower(),
            "light": "light" in shown.lower(),
            "entry": "door" in shown.lower() or "lock" in shown.lower(),
            "switch": True,
            "device": "tv",
            "to": "off" if "off" in request else "on",
            "each": True,
            "split": ", and " in request,
            "act": request.startswith("Turn"),
            "chat": "rain" in request and ", and " not in request,
        }.get(name, "preferably only one" in shown)

    def __call__(self, state, questions):
        answers = {}
        for name, question in questions.items():
            answer = self.answer(state["inputs"], name, question)
            if question["type"] == "noul":
                answers[name] = {"noul": 0.9 if answer else 0.1}
            else:
                answers[name] = {"probabilities": {label: float(label == str(answer)) for label in question["criteria"]}, "confidence": 1.0}
        return answers


COOL_TO_18 = "def cool_to_18():\n    while heat() > 18:\n        cooler()\n    return f'it is {heat()} degrees'"


def llm(model=None):
    """A stand-in generative model for the page's one generative example."""
    return DummyLM({
        "Turn the TV off, and tell me whether it will rain.": {"parts": ["Turn the TV off.", "Will it rain today?"]},
        "Will it rain today?": {"reply": "I cannot see the weather from here."},
        "Bring the heat down to 18 degrees.": {
            "reasoning": "Loop until it is 18, and submit the source.",
            "code": f"```python\nSUBMIT(wrote=define(name='cool_to_18', description='Cool the house until it is 18 degrees.', code={COOL_TO_18!r}))\n```",
        },
    })


def run(jev, generative, interpreter_factory=None):
    program = "\n".join(blocks())
    program = re.sub(r"\bTypeSafe\(", "_jev(", program).replace("dspy.LM(", "_llm(")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), dspy.context(interpreter_factory=interpreter_factory):
        exec(compile(program, str(PAGE), "exec"), {"_jev": jev, "_llm": generative, "__name__": "rdm_docs"})
    return printed.getvalue().splitlines()


def assert_prints_what_the_page_says(printed):
    lines = iter(printed)
    for line in documented(blocks()):
        assert any(line == got for got in lines), f"The page says the program prints {line!r}; it did not, in that order.\n" + "\n".join(printed)


def test_the_reference_page_prints_what_it_says():
    in_process = pytest.importorskip("dspy_interpreters").InProcessInterpreter
    assert len(documented(blocks())) >= 9
    assert_prints_what_the_page_says(run(Jev, llm, in_process))


@pytest.mark.llm_call
def test_the_reference_page_prints_what_it_says_with_jev():
    if not (os.environ.get("TYPESAFE_API_KEY") and os.environ.get("OPENROUTER_API_KEY")):
        pytest.skip("needs TYPESAFE_API_KEY and OPENROUTER_API_KEY")
    assert_prints_what_the_page_says(run(
        lambda model: TypeSafe(model, cache=False),
        lambda model: dspy.LM("openrouter/anthropic/claude-haiku-4.5", cache=False),
    ))
