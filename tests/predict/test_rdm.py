"""Tests for the RDM module.

Test organization:
- One request: answered once, and its answers select tools
- Sub-calls: a tool calls the module, and the answers come back to the tool
- What a call returns
- A map: an output typed as a tool points at a tool, turn after turn
- The leaf: at max_depth a call is a plain Predict
- Limits and errors
- Thresholds and ReAnchor
- Tools that are modules, and other backends
"""

import asyncio
import functools
import json
import threading

import pytest

import dspy
from dspy.experimental import RDM, Choice, Noul, ReAnchor, Score
from dspy.utils.dummies import DummyLM


class Client:
    """A stand-in System One client: `answer(inputs, name, question)` gives each question's answer."""

    supports_decision_requests = True
    cache = True

    def __init__(self, answer, tokens=0):
        self.answer = answer
        self.tokens = tokens
        self.model = "jev-test"
        self.calls = []
        self.history = []
        self.callbacks = []

    def __call__(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self.tokens and dspy.settings.usage_tracker:
            dspy.settings.usage_tracker.add_usage("jev-test", {"prompt_tokens": self.tokens, "completion_tokens": 0})
        return {name: self.evidence(question, self.answer(state["inputs"], name, question)) for name, question in questions.items()}

    @staticmethod
    def evidence(question, answer):
        """An answer as the probabilities a System One model returns for that type of question."""
        if question["type"] == "noul":
            return {"noul": {True: 0.9, False: 0.1}.get(answer, answer)}
        if isinstance(answer, dict):
            return {"probabilities": answer, "confidence": max(answer.values())}
        labels = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
        return {"probabilities": {label: float(label == str(answer)) for label in labels}, "confidence": 1.0}


def requests(result, **where):
    return [entry for entry in result.trajectory if all(entry.get(key) == value for key, value in where.items())]


Frustration = Score["calm", "frustrated", "angry"]
Action = Choice[("on", "turn them on"), ("off", "turn them off")]
Room = Choice[("kitchen", ""), ("bedroom", ""), ("house", "every room")]
Temperature = Choice[("warmer", ""), ("cooler", ""), (None, "The request is not about temperature.")]
Way = Choice[("north", ""), ("south", ""), ("east", "")]


# One request: answered once, and its answers select tools


class Triage(dspy.Signature):
    """Triage a support ticket."""

    ticket: str = dspy.InputField()
    refund: Noul = dspy.OutputField(desc="Is the customer asking for a refund?")
    escalate: Noul = dspy.OutputField(desc="Does the ticket need an engineer?")
    frustration: Frustration = dspy.OutputField(desc="How frustrated is the customer?")


def test_a_request_is_answered_once_and_an_output_named_after_a_tool_calls_it():
    called = []

    def refund(ticket):
        """Start a refund."""
        called.append(("refund", ticket))
        return "refund started"

    def escalate(ticket):
        """Page an engineer."""
        called.append(("escalate", ticket))

    client = Client(lambda inputs, name, question: {"refund": True, "escalate": False, "frustration": 1}[name])
    rdm = RDM(Triage, tools=[refund, escalate])
    rdm.set_lm(client)
    result = rdm(ticket="I was charged twice.")
    # One request: the state and every question, as a System One model takes them. Nothing is asked twice.
    assert len(client.calls) == 1 and client.calls[0]["state"]["inputs"] == {"ticket": "I was charged twice."}
    assert {name: question["type"] for name, question in client.calls[0]["questions"].items()} == {
        "refund": "noul", "escalate": "noul", "frustration": "score",
    }
    # The answers come back typed, with their probabilities, and the yes called its tool.
    assert (result.refund.value, result.refund.probability, result.escalate.value) == (True, 0.9, False)
    assert called == [("refund", "I was charged twice.")]
    # What the tool returned comes back to the caller, and the call says it did everything it selected.
    assert result.observations == {"refund": "refund started"} and result.termination_reason == "final"
    # A Score is a judgment and calls nothing, whatever it is named.
    assert result.frustration.value == 1.0
    assert result.trajectory == [{
        "depth": 0, "iteration": 0, "parent": None, "tool_name": None,
        "inputs": {"ticket": "I was charged twice."},
        "tokens": 0,
        "model": "jev-test",
        "outputs": {
            "refund": {"value": True, "confidence": result.refund.confidence, "probability": 0.9},
            "escalate": {"value": False, "confidence": result.escalate.confidence, "probability": 0.1},
            "frustration": {"value": 1.0, "confidence": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "level": 1},
        },
        "observations": {"refund": "refund started"},
    }]
    assert [name for name, _ in rdm.named_predictors()] == ["predict"] and list(rdm.tools) == ["refund", "escalate"]


def test_with_inputs_that_fit_it_is_predict():
    client = Client(lambda inputs, name, question: {"refund": 0.8, "escalate": 0.3, "frustration": 2}[name])
    rdm = RDM(Triage)
    with dspy.context(lm=client):
        ours, theirs = rdm(ticket="I was charged twice."), dspy.Predict(Triage)(ticket="I was charged twice.")
    # The same request, once, and the same answers: the module is a Predict until something does not fit.
    assert len(client.calls) == 2 and client.calls[0] == client.calls[1]
    assert all(ours[name] == theirs[name] for name in Triage.output_fields) and len(ours.trajectory) == 1


def test_arguments_come_from_the_other_answers():
    class Home(dspy.Signature):
        """Carry out a request to a smart home."""

        request: str = dspy.InputField()
        lights: Noul = dspy.OutputField(desc="Is the request a command to the lights?")
        action: Action = dspy.OutputField(desc="What should be done to the lights?")
        room: Room = dspy.OutputField(desc="Which lights does the request mean?")
        thermostat: Temperature = dspy.OutputField(desc="How should the temperature change?")

    done = []

    def lights(action, room):
        done.append(f"lights {action} in {room}")

    def thermostat(thermostat):
        done.append(f"thermostat {thermostat}")

    answers = {"lights": True, "action": "off", "room": "house", "thermostat": None}
    rdm = RDM(Home, tools=[lights, thermostat])
    client = Client(lambda inputs, name, question: answers[name])
    rdm.set_lm(client)
    result = rdm(request="Turn off all of the lights in the house")
    # Every question was asked at once, before it was known which would matter. The tool the
    # answers selected took its arguments from the answers of the same name.
    assert done == ["lights off in house"] and result.thermostat.value is None and len(client.calls) == 1
    # A Choice calls the tool of its name, which takes the chosen option by that name; choosing None calls nothing.
    answers.update(lights=False, thermostat="warmer")
    rdm(request="It is freezing in here")
    assert done == ["lights off in house", "thermostat warmer"]


def test_every_tool_that_is_selected_runs_and_they_run_at_once():
    class Next(dspy.Signature):
        """Decide how to look for an answer."""

        query: str = dspy.InputField()
        external_search: Noul = dspy.OutputField(desc="Would a web search help?")
        internal_search: Noul = dspy.OutputField(desc="Would the internal wiki help?")
        llm: Noul = dspy.OutputField(desc="Would rewording the query help?")
        child: Noul = dspy.OutputField(desc="Should the query be broken into parts?")

    barrier = threading.Barrier(4, timeout=10)
    threads = set()

    def make(name):
        def tool(query):
            # Each waits for the other three: this only returns if all four are running together.
            barrier.wait()
            threads.add(threading.get_ident())
            return f"{name} found nothing for {query}"

        return dspy.Tool(tool, name=name)

    names = ("external_search", "internal_search", "llm", "child")
    rdm = RDM(Next, tools=[make(name) for name in names])
    rdm.set_lm(Client(lambda inputs, name, question: True))
    result = rdm(query="q")
    # Four yeses, four calls, with no limit on how many may be made from one request.
    assert not barrier.broken and len(threads) == 4 and len(result.trajectory) == 1
    assert result.observations == {name: f"{name} found nothing for q" for name in names}


# Sub-calls: a tool calls the module, and the answers come back to the tool


class Find(dspy.Signature):
    """Find out whether an archive is worth searching for a question."""

    archive: dict = dspy.InputField()
    question: str = dspy.InputField()
    look: Noul = dspy.OutputField(desc="Could an archive of this kind answer `inputs.question`?")


class Judge(dspy.Signature):
    """Judge one record of an archive against a question."""

    record: str = dspy.InputField()
    question: str = dspy.InputField()
    answers: Noul = dspy.OutputField(desc="Does `inputs.record` answer `inputs.question`?")


def test_a_tool_calls_the_module_on_a_piece_and_the_answers_come_back_to_the_tool():
    archive = {"a": "The hangar opens at dawn.", "b": "The hangar door sticks in frost.", "c": "Tides turn at noon."}

    def look(question, archive):
        """Judge each record that mentions the hangar, in a sub-call of its own, and keep those that answer."""
        records = [text for text in archive.values() if "hangar" in text]
        verdicts = [rdm(signature=Judge, record=record, question=question) for record in records]
        return [record for record, verdict in zip(records, verdicts, strict=True) if verdict.answers.value]

    rdm = RDM(Find, tools=[look])
    client = Client(lambda inputs, name, question: name == "look" or "dawn" in inputs["record"])
    rdm.set_lm(client)
    result = rdm(archive=archive, question="When does the hangar open?")
    # The tool called the module twice. Each is a sub-call: one level deeper, with its own small
    # state and its own question, asked once, and recorded under the tool that made it.
    assert [(entry["depth"], entry["parent"], entry["tool_name"]) for entry in result.trajectory] == [
        (0, None, None), (1, 0, "look"), (1, 0, "look"),
    ]
    assert [call["state"]["inputs"] for call in client.calls[1:]] == [
        {"record": "The hangar opens at dawn.", "question": "When does the hangar open?"},
        {"record": "The hangar door sticks in frost.", "question": "When does the hangar open?"},
    ]
    assert [list(call["questions"]) for call in client.calls] == [["look"], ["answers"], ["answers"]]
    # Their answers came back to the tool, which decided in code what to make of them.
    assert result.observations == {"look": ["The hangar opens at dawn."]}
    assert [entry["outputs"]["answers"]["value"] for entry in requests(result, depth=1)] == [True, False]
    # A sub-call returns what any call returns, without the run's trajectory.
    with dspy.context(lm=client):
        alone = RDM(Judge)(record="x", question="q")
    assert alone.termination_reason == "final" and alone.observations == {} and len(alone.trajectory) == 1


def test_sub_calls_made_through_batch_join_the_run_too():
    archive = {str(n): f"Record {n} about the hangar." for n in range(6)}

    def look(question, archive):
        examples = [
            dspy.Example(signature=Judge, record=record, question=question).with_inputs("signature", "record", "question")
            for record in archive.values()
        ]
        return sum(bool(verdict.answers) for verdict in rdm.batch(examples, disable_progress_bar=True))

    rdm = RDM(Find, tools=[look], max_predictor_calls=5)
    rdm.set_lm(Client(lambda inputs, name, question: True))
    result = rdm(archive=archive, question="q")
    # Six sub-calls at once, on the run's own count of requests: one for the root, four answered, two refused.
    assert sorted(entry["depth"] for entry in result.trajectory) == [0, 1, 1, 1, 1, 1, 1]
    assert len(requests(result, skipped="max_predictor_calls")) == 2 and result.observations == {"look": 4}
    assert result.termination_reason == "max_predictor_calls"


# What a call returns


def test_a_call_returns_its_own_answers_and_what_its_tools_came_to():
    class Claim(dspy.Signature):
        """Decide whether a claim holds."""

        claim: str = dspy.InputField()
        holds: Noul = dspy.OutputField(desc="Is `inputs.claim` true?")
        ledger: Noul = dspy.OutputField(desc="Would the ledger settle `inputs.claim`?")

    class Checked(dspy.Signature):
        """Decide whether a claim holds, given an entry."""

        claim: str = dspy.InputField()
        entry: str = dspy.InputField()
        holds: Noul = dspy.OutputField(desc="Does `inputs.entry` show that `inputs.claim` is true?")

    def ledger(claim):
        return rdm(signature=Checked, claim=claim, entry="Paid in full on 3 May.")

    rdm = RDM(Claim, tools=[ledger])
    rdm.set_lm(Client(lambda inputs, name, question: name == "ledger" or "Paid" in inputs.get("entry", "")))
    result = rdm(claim="The invoice was paid.")
    # The first request could not tell, and the ledger's sub-call could. Each answer stays with the
    # request that gave it: a judgment made on less is not overwritten by one made on more.
    assert (result.holds.value, result.ledger.value) == (False, True)
    assert result.observations["ledger"].holds.value is True
    assert [(entry["depth"], entry["outputs"]["holds"]["value"]) for entry in result.trajectory] == [(0, False), (1, True)]


class Home(dspy.Signature):
    """Carry out a request to a smart home."""

    request: str = dspy.InputField()
    split: Noul = dspy.OutputField(desc="Does `inputs.request` ask for more than one distinct action?")
    lock: Noul = dspy.OutputField(desc="Does `inputs.request` ask only for a door to be locked?")
    chat: Noul = dspy.OutputField(desc="Is `inputs.request` only a question to answer in conversation?")


def home(inputs, name, question):
    """Decides as the smart home's decision model would, from the words of the request."""
    request = inputs["request"]
    several = " and " in request
    return {"split": several, "lock": "lock" in request.lower() and not several, "chat": request.endswith("?") and not several}[name]


def test_a_tool_can_put_each_piece_of_a_request_to_the_module_one_level_down():
    done = []

    def split(request, call: RDM.Call):
        """What a generative model would write: the request's pieces. Each is a request of its own."""
        return [call.rdm(request=piece) for piece in request.split(" and ")]

    def lock(request):
        done.append(request)
        return "locked"

    def chat(request):
        return "It will rain."

    client = Client(home)
    rdm = RDM(Home, tools=[split, lock, chat])
    rdm.set_lm(client)
    result = rdm(request="Lock the front door and will it rain? and lock the shed")
    # The decision model decided once that the request has several parts, the tool made three
    # states, and each was decided by the same questions one level down. The tool chose nothing.
    assert [(entry["depth"], entry["parent"], entry["tool_name"]) for entry in result.trajectory] == [
        (0, None, None), (1, 0, "split"), (1, 0, "split"), (1, 0, "split"),
    ]
    assert [list(call["questions"]) for call in client.calls] == [["split", "lock", "chat"]] * 4
    assert [entry["inputs"]["request"] for entry in result.trajectory[1:]] == ["Lock the front door", "will it rain?", "lock the shed"]
    # The tool comes to what its pieces came to, in order, and each of those is what any call returns.
    made = result.observations["split"]
    assert [child.observations for child in made] == [{"lock": "locked"}, {"chat": "It will rain."}, {"lock": "locked"}]
    assert done == ["Lock the front door", "lock the shed"] and result.termination_reason == "final"


def test_what_a_tool_uses_of_a_model_of_its_own_is_counted_with_the_run():
    def split(request, call: RDM.Call):
        """A generative model writes the pieces, and reports what it used."""
        dspy.settings.usage_tracker.add_usage("writer", {"prompt_tokens": 70, "completion_tokens": 30})
        return [call.rdm(request=piece) for piece in request.split(" and ")]

    rdm = RDM(Home, tools=[split, dspy.Tool(lambda request: "locked", name="lock")])
    rdm.set_lm(Client(home, tokens=10))
    result = rdm(request="lock the gate and lock the shed")
    # The request's own tokens and the tool's are recorded apart, on the request whose answer called the tool.
    assert (result.trajectory[0]["tokens"], result.trajectory[0]["tool_tokens"]) == (10, {"split": 100})
    assert {entry["model"] for entry in result.trajectory} == {"jev-test"}
    assert all("tool_tokens" not in entry and entry["tokens"] == 10 for entry in result.trajectory[1:])

    # Both count against max_tokens: here the tool's use leaves nothing for the next turn.
    class Steps(dspy.Signature):
        """Take steps."""

        task: str = dspy.InputField()
        observation: str = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="What next?")

    def write():
        dspy.settings.usage_tracker.add_usage("writer", {"prompt_tokens": 100, "completion_tokens": 0})
        return "written"

    limited = RDM(Steps, tools=[write], max_tokens=100)
    limited.set_lm(Client(lambda inputs, name, question: "write", tokens=10))
    result = limited(task="t")
    assert [entry.get("skipped") for entry in result.trajectory] == [None, "max_tokens"]
    assert result.termination_reason == "max_tokens"


def test_what_a_request_uses_is_counted_once_however_deep_the_tool_that_asked_it():
    class Count(dspy.Signature):
        """Count."""

        n: int = dspy.InputField()
        again: Noul = dspy.OutputField(desc="Again?")

    def again(n, call: RDM.Call):
        """Three sub-calls, each of which makes one more, and a model of the tool's own at the top."""
        if call.depth == 0:
            dspy.settings.usage_tracker.add_usage("writer", {"prompt_tokens": 7, "completion_tokens": 0})
        return [call.rdm(n=10 * n + i + 1).again.value for i in range(3 if n == 0 else 1)]

    rdm = RDM(Count, tools=[again], max_depth=2, max_tokens=707)
    rdm.set_lm(Client(lambda inputs, name, question: True, tokens=100))
    with dspy.track_usage() as usage:
        result = rdm(n=0)
    # Seven requests of 100 tokens. A tool's tokens are what its own model used, not what the requests it asked used.
    assert [entry["tokens"] for entry in result.trajectory] == [100] * 7 and result.termination_reason == "max_depth"
    assert [entry.get("tool_tokens") for entry in result.trajectory] == [{"again": 7}] + [None] * 6
    # Whatever tracks usage around the run sees each use once as well.
    assert usage.get_total_tokens() == {
        "jev-test": {"prompt_tokens": 700, "completion_tokens": 0},
        "writer": {"prompt_tokens": 7, "completion_tokens": 0},
    }


def test_a_request_already_asked_on_the_way_is_not_asked_again():
    class Wait(dspy.Signature):
        """Wait for a door to open."""

        door: str = dspy.InputField()
        wait: Noul = dspy.OutputField(desc="Is `inputs.door` still shut?")

    def wait(door, call: RDM.Call):
        """Changes nothing, and asks the same question again."""
        return call.rdm(door=door)

    # Asked again it would get the same answer and select the same tool, so it is not asked: the
    # run has come round, and says so.
    rdm = RDM(Wait, tools=[wait])
    client = Client(lambda inputs, name, question: True)
    rdm.set_lm(client)
    result = rdm(door="shut")
    assert len(client.calls) == 1 and [(entry["depth"], "repeat" in entry) for entry in result.trajectory] == [(0, False), (1, True)]
    assert result.termination_reason == "repeat" and result.observations["wait"].wait.value is True

    # The same holds further down: a state that comes back as its own descendant is a cycle.
    def split(request, call: RDM.Call):
        return [call.rdm(request="lock the gate and lock the shed"), call.rdm(request="lock the shed")]

    rdm = RDM(Home, tools=[split, dspy.Tool(lambda request: "locked", name="lock")], max_depth=9)
    client = Client(home)
    rdm.set_lm(client)
    result = rdm(request="lock the gate and lock the shed")
    assert [(entry["depth"], "repeat" in entry) for entry in result.trajectory] == [(0, False), (1, True), (1, False)]
    assert len(client.calls) == 2 and result.termination_reason == "repeat"

    # When a limit stopped something as well, the limit is the reason: it bounds more than a repeat does.
    def stay_and_knock(door, call: RDM.Call):
        call.rdm(door="another door")
        return call.rdm(door=door)

    rdm = RDM(Wait, tools=[dspy.Tool(stay_and_knock, name="wait")], max_predictor_calls=1)
    rdm.set_lm(Client(lambda inputs, name, question: True))
    result = rdm(door="shut")
    assert [(entry.get("skipped"), entry.get("repeat")) for entry in result.trajectory] == [
        (None, None), ("max_predictor_calls", None), (None, True),
    ]
    assert result.termination_reason == "max_predictor_calls"
    # Two states that are alike but not on each other's way are both asked.
    rdm = RDM(Home, tools=[dspy.Tool(lambda request: [rdm(request="lock up"), rdm(request="lock up")], name="split")])
    rdm.set_lm(Client(lambda inputs, name, question: name == "split" and inputs["request"] == "twice"))
    result = rdm(request="twice")
    assert ["repeat" in entry for entry in result.trajectory] == [False, False, False] and result.termination_reason == "final"

    # The entries of a dict are part of what a request shows, in the order given and whatever their keys.
    rdm = RDM(dspy.Signature({"door": (dict, dspy.InputField()), "wait": (Noul, dspy.OutputField(desc="Is any door shut?"))}))
    rdm.set_lm(Client(lambda inputs, name, question: True))
    assert rdm(door={1: "front", "b": "back"}).wait.value is True


def test_fit_is_given_each_request_before_it_is_asked():
    class Length(dspy.Signature):
        """Judge the length of a ticket."""

        ticket_length: int = dspy.InputField()
        long: Noul = dspy.OutputField(desc="Is `inputs.ticket_length` more than forty?")

    seen = []

    def fit(signature, inputs, call):
        """Shows the ticket's first line only, after asking a question of its own about the whole of it."""
        seen.append((list(signature.output_fields), call.depth))
        if "ticket" not in inputs or "\n" not in inputs["ticket"]:
            return signature, inputs
        long = call.rdm(signature=Length, ticket_length=len(inputs["ticket"]), max_depth=0).long
        return signature, {**inputs, "ticket": inputs["ticket"].splitlines()[0] + (" [...]" if long else "")}

    called = {}

    def refund(ticket, call: RDM.Call):
        called.update(ticket=ticket, whole=call.inputs["ticket"])

    client = Client(lambda inputs, name, question: {"refund": True, "escalate": False, "frustration": 0, "long": True}[name])
    rdm = RDM(Triage, tools=[refund], fit=fit)
    rdm.set_lm(client)
    result = rdm(ticket="I was charged twice.\nOrder 98423, last Thursday.")
    # What fit returned is what the request showed. What it asked first is a sub-call of that request.
    assert [call["state"]["inputs"] for call in client.calls] == [{"ticket_length": 48}, {"ticket": "I was charged twice. [...]"}]
    assert [(entry["depth"], entry["parent"], list(entry["outputs"])) for entry in result.trajectory] == [
        (0, None, ["refund", "escalate", "frustration"]), (1, 0, ["long"]),
    ]
    # It is given every request, its own included. A tool is given the call's inputs whole: fit
    # changes what the model is shown, not what the code acts on.
    assert seen == [(["refund", "escalate", "frustration"], 0), (["long"], 1)]
    whole = "I was charged twice.\nOrder 98423, last Thursday."
    assert called == {"ticket": whole, "whole": whole}

    # What a fit uses on a model of its own is counted with the run, on the request it was fitting.
    def rewrite(signature, inputs, call):
        dspy.settings.usage_tracker.add_usage("writer", {"prompt_tokens": 70, "completion_tokens": 0})
        return signature, inputs

    rdm = RDM(Triage, fit=rewrite, max_tokens=170)
    rdm.set_lm(Client(lambda inputs, name, question: {"refund": False, "escalate": False, "frustration": 0}[name], tokens=100))
    with dspy.track_usage() as usage:
        result = rdm(ticket="Thanks!")
    assert (result.trajectory[0]["tokens"], result.trajectory[0]["fit_tokens"]) == (100, 70)
    assert {model: used["prompt_tokens"] for model, used in usage.get_total_tokens().items()} == {"writer": 70, "jev-test": 100}


TAXONOMY = {
    "Animals & Pet Supplies": {"Cat Supplies": {"Cat Beds": {}, "Cat Window Perches": {}}, "Bird Supplies": {"Bird Perches": {}}},
    "Furniture": {"Beds": {"Bunk Beds": {}}, "Shelving": {"Wall Shelves": {}, "Bookcases": {}}},
}


def children(path):
    node = TAXONOMY
    for label in path:
        node = node[label]
    return node


def ask(path):
    """The question at one node of the taxonomy: its options are that node's children, known only once it is reached."""
    options = Choice[tuple((label, "") for label in children(path))]
    return dspy.Signature(
        {
            "listing": (str, dspy.InputField()),
            "path": (list, dspy.InputField(desc="The categories chosen so far.")),
            "descend": (options, dspy.OutputField(desc="Which direct child category best matches the listing?")),
        },
        "Classify a product listing.",
    )


def descend(descend, path, listing, call: RDM.Call):
    """Go into the chosen category and ask about its own children, one level down. Comes to the path to a leaf."""
    path = [*path, descend]
    return call.rdm(signature=ask(path), listing=listing, path=path).observations["descend"] if children(path) else path


def pick(inputs, name, question):
    """Chooses the option that shares the most words with the listing."""
    words = set(inputs["listing"].lower().split())
    return max(question["criteria"], key=lambda label: len(words & set(label.lower().replace("&", "").split())))


LISTING = "A padded cat window perch with suction cups, in the pet supplies aisle."


def test_a_decision_can_be_about_which_decision_comes_next():
    rdm = RDM(ask([]), tools=[descend])
    client = Client(pick)
    rdm.set_lm(client)
    result = rdm(listing=LISTING, path=[])
    # Three requests, each with an answer space that did not exist until the one before was answered.
    assert [list(call["questions"]["descend"]["criteria"]) for call in client.calls] == [
        ["Animals & Pet Supplies", "Furniture"], ["Cat Supplies", "Bird Supplies"], ["Cat Beds", "Cat Window Perches"],
    ]
    # Each is a sub-call of the one before, and carries the path so far: state the tool built.
    assert [(entry["depth"], entry["parent"]) for entry in result.trajectory] == [(0, None), (1, 0), (2, 1)]
    assert [call["state"]["inputs"]["path"] for call in client.calls] == [
        [], ["Animals & Pet Supplies"], ["Animals & Pet Supplies", "Cat Supplies"],
    ]
    # The call's own answer is its own choice. What the tool came to is where the descent ended.
    assert result.descend.value == "Animals & Pet Supplies"
    assert result.observations["descend"] == ["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"]


def test_a_tool_must_be_able_to_take_its_arguments():
    # A parameter that names no answer or input is refused, even one alone: a misspelt name, or a
    # parameter written for another module, never quietly receives something else.
    for name, tool in (("refund", lambda tickt: None), ("refund", lambda first, second: None)):
        careless = RDM(Triage, tools=[dspy.Tool(tool, name=name)])
        careless.set_lm(Client(lambda inputs, name, question: True if question["type"] == "noul" else 0))
        with pytest.raises(ValueError, match=r"Tool 'refund' takes \[.*\], which name no answer or input"):
            careless(ticket="t")
    # A parameter with a default may be left to it.
    rdm = RDM(Triage, tools=[dspy.Tool(lambda ticket, note="none": note, name="refund")])
    rdm.set_lm(Client(lambda inputs, name, question: True if question["type"] == "noul" else 0))
    assert rdm(ticket="t").observations["refund"] == "none"

    # A function that takes whatever it is given says what it takes on the tool, as a tool made from
    # an MCP server does. What has the name of an answer or an input is given, and the rest left out.
    def remote(**arguments):
        return arguments

    declared = {"ticket": {"type": "string"}, "note": {"type": "string"}}
    rdm = RDM(Triage, tools=[dspy.Tool(remote, name="refund", desc="Start a refund.", args=declared)])
    rdm.set_lm(Client(lambda inputs, name, question: True if question["type"] == "noul" else 0))
    assert rdm(ticket="t").observations["refund"] == {"ticket": "t"}


def test_what_cannot_be_used_is_refused_when_the_module_is_built_or_called():
    for signature in ("state, question -> found: bool", "question -> trajectory: bool", "question -> observations: bool"):
        with pytest.raises(ValueError, match="reserved by RDM"):
            RDM(signature)
    with pytest.raises(TypeError, match="tools must be a list"):
        RDM(Triage, tools={"refund": print})
    with pytest.raises(ValueError, match="Duplicate tool name"):
        RDM(Triage, tools=[descend, descend])
    with pytest.raises(ValueError, match="at least 1"):
        RDM(Triage, max_predictor_calls=0)
    rdm = RDM(Triage)
    with pytest.raises(ValueError, match="The call has inputs \\[\\]"):
        rdm()
    with pytest.raises(ValueError, match="The call has inputs \\['extra', 'ticket'\\]"):
        rdm(ticket="t", extra=1)


def test_a_tool_that_takes_the_call_reads_the_run_state_and_the_module():
    seen, calls = {}, []

    def look(question, call: RDM.Call):
        # The module is on the call, so a tool needs no reference of its own to make a sub-call.
        seen.update(call.state)
        calls.append(call)
        hits = call.state["index"][call.state["run_id"]]
        return [call.rdm(signature=Judge, record=record, question=question).answers.value for record in hits]

    rdm = RDM(Find, tools=[look])
    client = Client(lambda inputs, name, question: True)
    rdm.set_lm(client)
    index = {"run 7": ["The hangar opens at dawn."]}
    result = rdm(archive={"a": "x"}, question="q", state={"index": index, "run_id": "run 7", "question": "not an input"})
    # What was passed as `state` reached the tool and no request: it is state the model never sees.
    # Its names are its own, so one may be the name of an input.
    assert seen == {"index": index, "run_id": "run 7", "question": "not an input"} and result.observations == {"look": [True]}
    assert "run 7" not in json.dumps(client.calls) and "index" not in json.dumps(client.calls[0])
    assert [entry["depth"] for entry in result.trajectory] == [0, 1]
    # The call says where the tool is: the depth, the turn, what the request showed, the history so far, and the requests left.
    call = calls[0]
    assert (call.depth, call.iteration, call.inputs, len(call.trajectory), call.requests_left) == (0, 0, {"archive": {"a": "x"}, "question": "q"}, 1, 255)

    # A module that is given state works the same when it is called inside another module's tool:
    # the state is its own and that of the calls under it, and the caller's is as it was.
    inner_seen, outer_seen = [], []

    def lookup(question, call: RDM.Call):
        inner_seen.append(dict(call.state))
        return call.rdm(signature=Judge, record=call.state["shelf"], question=question, max_depth=0).answers.value

    inner = RDM(Find, tools=[dspy.Tool(lookup, name="look")])

    def outer_look(question, archive, call: RDM.Call):
        outer_seen.append(dict(call.state))
        found = inner(archive=archive, question=question, state={"shelf": "The hangar opens at dawn."}).observations["look"]
        outer_seen.append(dict(call.state))
        return found

    outer = RDM(Find, tools=[dspy.Tool(outer_look, name="look")])
    for module in (inner, outer):
        module.set_lm(client)
    assert inner(archive={}, question="q", state={"shelf": "alone"}).observations == {"look": True}
    assert outer(archive={}, question="q", state={"who": "outer"}).observations == {"look": True}
    assert inner_seen == [{"shelf": "alone"}, {"shelf": "The hangar opens at dawn."}] and outer_seen == [{"who": "outer"}] * 2


# A map: an output typed as a tool points at one of the module's tools, turn after turn


class House:
    """A house that pushes back: it will not lock an open door."""

    def __init__(self):
        self.devices = {"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}

    def set(self, device, state):
        if device == "lock" and state == "locked" and self.devices["door"] == "open":
            return "lock: refused, the door is open"
        self.devices[device] = state
        return f"{device}: now {state}"


class Operate(dspy.Signature):
    """Operate a smart home, one step at a time."""

    request: str = dspy.InputField()
    devices: dict = dspy.InputField(desc="Every device and its state now.")
    observation: str = dspy.InputField(desc="What the last step did, if there was one.")
    next: dspy.Tool = dspy.OutputField(desc="What should happen next for `inputs.request`?")


def lines(house):
    """Every step the house can take, each a tool with a name and a description."""
    steps = {"tv on": ("tv", "on"), "lights off": ("lights", "off"), "close door": ("door", "closed"), "lock door": ("lock", "locked")}
    return [
        dspy.Tool(functools.partial(house.set, device, state), name=name, desc=f"Set the {device} to {state}.")
        for name, (device, state) in steps.items()
    ]


def scripted(*names):
    """A decision model that points at these names, in order."""
    script = iter(names)
    return Client(lambda inputs, name, question: next(script))


def test_something_that_offers_tools_is_given_whole_and_its_tools_are_on_the_map_as_they_are_then():
    class Shelf(dspy.Module):
        """Something that offers tools of its own, which change while a run goes on."""

        def __init__(self):
            super().__init__()
            self.names = ["first"]

        @property
        def offers(self):
            return [dspy.Tool(self.more if name == "first" else (lambda: "did second"), name=name, desc=f"The {name}.") for name in self.names]

        def more(self):
            self.names = ["second"]
            return "put up the second"

    shelf = Shelf()
    rdm = RDM(Operate, tools=[*lines(House()), shelf])
    client = scripted("first", "second", "finish")
    rdm.set_lm(client)
    result = rdm(request="Go on.", devices={})
    # Each request's map holds what the shelf offers then: a name it puts up is offered from the next
    # request, and one it takes down is gone. Its tools are not copied into `rdm.tools`.
    offered = [list(call["questions"]["next"]["criteria"])[4:] for call in client.calls]
    assert offered == [["first", "finish"], ["second", "finish"], ["second", "finish"]] and "first" not in rdm.tools
    assert result.observations["next"] == [["first", "put up the second"], ["second", "did second"], ["finish", None]]
    # It is part of the module: reached where the module's parts are, and a copy of the module has a copy of it.
    assert rdm.sources == [shelf] and rdm.deepcopy().sources[0] is not shelf
    # A name it offers that the map already has is a mistake of the program.
    shelf.names = ["tv on"]
    with pytest.raises(ValueError, match="\\['tv on'\\] are offered by `Shelf` and are already names on the map"):
        rdm(request="Go on.", devices={})


def test_an_output_typed_as_a_tool_points_at_a_tool_turn_after_turn_until_finish():
    house = House()
    rdm = RDM(Operate, tools=lines(house))
    client = scripted("tv on", "lights off", "finish")
    rdm.set_lm(client)
    result = rdm(request="Movie time.", devices=house.devices)
    # Each request offers the module's tools as options, each with its description, and `finish`.
    criteria = client.calls[0]["questions"]["next"]["criteria"]
    assert list(criteria) == ["tv on", "lights off", "close door", "lock door", "finish"]
    assert criteria["tv on"] == "Set the tv to on."
    # The tool pointed at runs, and the same question is asked again about the house as it is now.
    assert [call["state"]["inputs"]["devices"]["tv"] for call in client.calls] == ["off", "on", "on"]
    assert [call["state"]["inputs"]["observation"] for call in client.calls] == ["", "tv: now on", "lights: now off"]
    assert house.devices["tv"] == "on" and house.devices["lights"] == "off"
    # What was done, in order, and every turn at the same level.
    assert result.observations["next"] == [["tv on", "tv: now on"], ["lights off", "lights: now off"], ["finish", None]]
    assert [(entry["depth"], entry["iteration"], entry["outputs"]["next"]["value"]) for entry in result.trajectory] == [
        (0, 0, "tv on"), (0, 1, "lights off"), (0, 2, "finish"),
    ]
    # The trajectory holds what each request showed, though the tools changed the same dict since.
    assert [entry["inputs"]["devices"]["tv"] for entry in result.trajectory] == ["off", "on", "on"]
    # The call's own answer is its first point.
    assert result.next.value == "tv on" and result.termination_reason == "final"


def test_the_observation_shows_what_was_refused_so_the_next_point_can_put_it_right():
    house = House()
    rdm = RDM(Operate, tools=lines(house))
    client = scripted("lock door", "close door", "lock door", "finish")
    rdm.set_lm(client)
    result = rdm(request="Lock up.", devices=house.devices)
    assert client.calls[1]["state"]["inputs"]["observation"] == "lock: refused, the door is open"
    assert house.devices == {"tv": "off", "lights": "on", "door": "closed", "lock": "locked"}
    assert [name for name, _ in result.observations["next"]] == ["lock door", "close door", "lock door", "finish"]


def test_a_tool_added_while_the_module_runs_is_offered_from_the_next_request():
    house = House()
    tools = lines(house)
    movie = dspy.Tool(lambda: [tool() for tool in tools if tool.name in ("tv on", "lights off")] and "tv: now on, lights: now off",
                      name="movie time", desc="Do again what was done for 'Movie time.'.")

    def keep(call: RDM.Call):
        """Keep what was just done as one tool: for the rest of this run, and for the runs after it."""
        call.tools["movie time"] = rdm.tools["movie time"] = movie
        return "kept as movie time"

    rdm = RDM(Operate, tools=[*tools, keep])
    client = scripted("tv on", "lights off", "keep", "finish", "movie time", "finish")
    rdm.set_lm(client)
    rdm(request="Movie time.", devices=house.devices)
    assert "movie time" not in client.calls[2]["questions"]["next"]["criteria"]
    assert "movie time" in client.calls[3]["questions"]["next"]["criteria"]
    # The next run is offered it, and one point does what two did.
    house.devices.update(tv="off", lights="on")
    result = rdm(request="Let's watch a film.", devices=house.devices)
    assert result.observations["next"] == [["movie time", "tv: now on, lights: now off"], ["finish", None]]
    assert house.devices["tv"] == "on" and house.devices["lights"] == "off"


def test_each_run_points_at_its_own_copy_of_the_tools():
    # A tool put in `call.tools` is this run's: another run, here at the same time on another
    # thread, is not offered it, and the module's own tools are as they were.
    def remember(request, call: RDM.Call):
        call.tools[f"redo {request}"] = dspy.Tool(lambda: "redone", name=f"redo {request}", desc=f"Do '{request}' again.")
        return "remembered"

    class Remember(dspy.Signature):
        """Remember requests."""

        request: str = dspy.InputField()
        observation: str = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="What next?")

    barrier = threading.Barrier(2, timeout=10)

    def answer(inputs, name, question):
        if inputs["observation"] == "":
            return "remember"
        barrier.wait()  # both runs have remembered before either is asked again
        return "finish"

    client = Client(answer)
    rdm = RDM(Remember, tools=[remember])
    rdm.set_lm(client)
    rdm.batch([{"request": "a"}, {"request": "b"}], num_threads=2, disable_progress_bar=True)
    offered = [set(call["questions"]["next"]["criteria"]) for call in client.calls if call["state"]["inputs"]["observation"]]
    assert sorted(offered, key=sorted) == [{"remember", "redo a", "finish"}, {"remember", "redo b", "finish"}]
    assert list(rdm.tools) == ["remember"]


def test_a_long_loop_is_a_loop_and_its_end_is_recorded():
    class Loop(dspy.Signature):
        """Count."""

        count: dict = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="What next?")

    def step(count):
        count["n"] += 1
        return count["n"]

    # Five hundred turns at one level: no deeper in Python than one.
    rdm = RDM(Loop, tools=[step], max_iters=500, max_predictor_calls=1000)
    rdm.set_lm(Client(lambda inputs, name, question: "step"))
    result = rdm(count={"n": 0})
    assert len(result.observations["next"]) == 500 and result.termination_reason == "max_iters"
    # Pointing at `finish` ends a loop, and says so where the tools are.
    rdm.set_lm(Client(lambda inputs, name, question: "finish" if inputs["count"]["n"] >= 2 else "step"))
    result = rdm(count={"n": 0})
    assert result.observations["next"] == [["step", 1], ["step", 2], ["finish", None]] and result.termination_reason == "final"
    # Two outputs that point are refused: each loop would put the other's question again.
    two = dspy.Signature({"count": (dict, dspy.InputField()), "a": (dspy.Tool, dspy.OutputField(desc="?")), "b": (dspy.Tool, dspy.OutputField(desc="?"))})
    with pytest.raises(ValueError, match="one output that points at a tool"):
        RDM(two, tools=[step])(count={"n": 0})


def test_a_tool_observed_is_shown_as_json_and_a_plain_function_on_the_map_is_a_tool():
    class Look(dspy.Signature):
        """Look."""

        observation: str = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="What next?")

    def rooms():
        return {"ok": True, "rooms": ["hall", None]}

    client = Client(lambda inputs, name, question: "finish" if inputs["observation"] else "rooms")
    rdm = RDM(Look, tools=[])
    rdm.tools["rooms"] = rooms  # a plain function: offered and run under its key
    rdm.set_lm(client)
    rdm()
    assert client.calls[1]["state"]["inputs"]["observation"] == '{"ok": true, "rooms": ["hall", null]}'



def test_a_turn_that_changes_nothing_ends_the_loop_as_a_repeat():
    house = House()
    rdm = RDM(Operate, tools=lines(house))
    client = scripted("tv on", "tv on", "tv on")
    rdm.set_lm(client)
    result = rdm(request="Put the TV on.", devices=house.devices)
    # The second turn shows the same house and the same observation as the third would: it is not asked again.
    assert len(client.calls) == 2 and result.termination_reason == "repeat"
    assert [name for name, _ in result.observations["next"]] == ["tv on", "tv on"]


def test_a_tool_named_finish_takes_the_place_of_the_built_in_one():
    house = House()
    said = []

    def finish():
        """Stop: everything the request asks for is so."""
        said.append("done")
        return "done"

    rdm = RDM(Operate, tools=[*lines(house), finish])
    client = scripted("tv on", "finish")
    rdm.set_lm(client)
    result = rdm(request="Put the TV on.", devices=house.devices)
    assert client.calls[0]["questions"]["next"]["criteria"]["finish"] == "Stop: everything the request asks for is so."
    assert list(client.calls[0]["questions"]["next"]["criteria"]).count("finish") == 1
    assert said == ["done"] and result.observations["next"] == [["tv on", "tv: now on"], ["finish", "done"]]
    assert len(client.calls) == 2


def test_a_tool_that_fails_is_shown_as_the_observation_and_the_loop_goes_on():
    house = House()

    def jam():
        """Jam the lock."""
        raise RuntimeError("the lock is jammed")

    rdm = RDM(Operate, tools=[*lines(house), jam])
    client = scripted("jam", "finish")
    rdm.set_lm(client)
    result = rdm(request="Jam it.", devices=house.devices)
    assert "the lock is jammed" in client.calls[1]["state"]["inputs"]["observation"]
    assert result.observations["next"][0][0] == "jam" and result.termination_reason == "final"


# The leaf: at max_depth a call is a plain Predict


class Count(dspy.Signature):
    """Count upwards."""

    n: int = dspy.InputField()
    again: bool = dspy.OutputField(desc="Should the counting go on?")


def test_the_module_calls_itself_until_max_depth_where_a_call_is_a_plain_predict():
    ran = []
    holder = {}

    def again(n):
        """Put the same question about the next two numbers, each in a sub-call."""
        ran.append(n)
        rdm = holder["rdm"]
        return [rdm(n=2 * n).termination_reason, rdm(n=2 * n + 1).termination_reason]

    def tree(max_depth):
        ran.clear()
        holder["rdm"] = RDM(Count, tools=[again], max_depth=max_depth)
        client = Client(lambda inputs, name, question: True)
        holder["rdm"].set_lm(client)
        return holder["rdm"](n=1), client

    # rdm > rdm > predict. The same module and the same question at every level; at depth 2 the
    # question is still answered, as by a plain Predict, and the tool it selected is not run.
    result, client = tree(2)
    assert sorted(entry["depth"] for entry in result.trajectory) == [0, 1, 1, 2, 2, 2, 2] and len(client.calls) == 7
    assert sorted(entry["inputs"]["n"] for entry in requests(result, depth=2)) == [4, 5, 6, 7]
    assert all(entry["outputs"]["again"] is True and entry["observations"] == {} for entry in requests(result, depth=2))
    assert sorted(ran) == [1, 2, 3] and result.observations == {"again": ["max_depth", "max_depth"]}
    # What the depth limit stopped two levels down is reported at every level above it.
    assert result.termination_reason == "max_depth"
    assert result.trajectory[1]["observations"] == {"again": ["max_depth", "max_depth"]}
    # With a depth of 1 the root uses its tools and every sub-call is a leaf.
    result, client = tree(1)
    assert len(client.calls) == 3 and ran == [1] and result.observations == {"again": ["max_depth", "max_depth"]}
    # With a depth of 0 the module itself is a plain Predict.
    result, client = tree(0)
    assert len(client.calls) == 1 and ran == [] and (result.again, result.termination_reason) == (True, "max_depth")


def test_a_choice_with_one_option_is_answered_without_being_asked():
    def ways(left, lit=False):
        fields = {
            "left": (list, dspy.InputField(desc="Doors not yet tried.")),
            "go": (Choice[tuple((door, "") for door in left)], dspy.OutputField(desc="Which door in `inputs.left` should be tried next?")),
        }
        if lit:
            fields["lit"] = (Noul, dspy.OutputField(desc="Is the room lit?"))
        return dspy.Signature(fields, "Leave the room.")

    tried = []

    def go(go, left, call: RDM.Call):
        tried.append(go)
        rest = [other for other in left if other != go]
        return call.rdm(signature=ways(rest), left=rest).observations["go"] if rest else "out"

    client = Client(lambda inputs, name, question: True if name == "lit" else list(question["criteria"])[-1])
    rdm = RDM(ways(["a", "b", "c"]), tools=[go], max_predictor_calls=2)
    rdm.set_lm(client)
    result = rdm(left=["a", "b", "c"])
    # Three choices, and two requests: with one door left there is nothing to decide. The answer is
    # that door, as certain, its tool runs like any other, and no limit is charged for it.
    assert tried == ["c", "b", "a"] and len(client.calls) == 2 and result.termination_reason == "final"
    last = result.trajectory[-1]
    assert (last["unasked"], last["tokens"], last["outputs"]["go"]["value"], last["outputs"]["go"]["probabilities"]) == (["go"], 0, "a", {"a": 1.0})
    assert result.observations["go"] == "out" and "unasked" not in result.trajectory[0]
    # Beside other questions, only the others are asked.
    client.calls.clear()
    rdm = RDM(ways(["a"], lit=True), tools=[dspy.Tool(lambda go, left: "out", name="go")])
    rdm.set_lm(client)
    result = rdm(left=["a"])
    assert [list(call["questions"]) for call in client.calls] == [["lit"]] and (result.go.value, result.lit.value) == ("a", True)
    assert list(result.trajectory[0]["outputs"]) == ["go", "lit"] and result.observations == {"go": "out"}


def test_a_call_can_be_asked_for_its_answers_only():
    ran = []

    def again(n, call: RDM.Call):
        ran.append(n)
        # The same question about another number, for its answer only: it says yes, and that runs nothing.
        leaf = call.rdm(n=n + 1, max_depth=0)
        # And about another, which may run its tool, but nothing below it may.
        deeper = call.rdm(n=n + 10, max_depth=1)
        assert (leaf.again, leaf.termination_reason, deeper.again) == (True, "final", True)
        return call.rdm(n=n + 100, max_depth=0)

    rdm = RDM(Count, tools=[again], max_depth=5)
    client = Client(lambda inputs, name, question: True)
    rdm.set_lm(client)
    result = rdm(n=1)
    # Every request said yes. The tool ran only where the call was allowed to go further.
    assert ran == [1, 11] and [entry["inputs"]["n"] for entry in result.trajectory] == [1, 2, 11, 12, 21, 111, 101]
    assert [entry["depth"] for entry in result.trajectory] == [0, 1, 1, 2, 2, 2, 1]
    # Asking for answers only keeps nothing from being done, so it is not a limit that was reached.
    # The limit put on the deeper call did stop a tool, and that is reported all the way up.
    assert result.observations["again"].termination_reason == "final" and result.termination_reason == "max_depth"
    # The call that starts a run can be asked the same way: it is then a Predict.
    ran.clear()
    result = rdm(n=1, max_depth=0)
    assert (ran, len(result.trajectory), result.again, result.termination_reason) == ([], 1, True, "final")


# Limits and errors


class Tick(dspy.Signature):
    """Count by pointing."""

    counter: dict = dspy.InputField()
    next: dspy.Tool = dspy.OutputField(desc="What next?")


class Begin(dspy.Signature):
    """Decide whether to count."""

    counter: dict = dspy.InputField()
    count: Noul = dspy.OutputField(desc="Should the counting start?")


def tick(counter):
    counter["n"] += 1
    return counter["n"]


def test_another_rdm_called_from_a_tool_joins_the_run_and_keeps_to_its_own_limits_too():
    class Start(dspy.Signature):
        """Decide whether to count."""

        n: int = dspy.InputField()
        count: Noul = dspy.OutputField(desc="Should the counting start?")

    ran, holder = [], {}

    def again(n):
        ran.append(n)
        return holder["inner"](n=n + 1).termination_reason

    def run(outer_depth, inner_depth):
        ran.clear()
        holder["inner"] = inner = RDM(Count, tools=[again], max_depth=inner_depth)
        outer = RDM(Start, tools=[dspy.Tool(inner, name="count")], max_depth=outer_depth)
        client = Client(lambda inputs, name, question: True)
        inner.set_lm(client)
        outer.set_lm(client)
        return outer(n=1)

    # The other module is a sub-call of the run it was called in: one trajectory.
    result = run(outer_depth=9, inner_depth=2)
    assert [(entry["depth"], entry["inputs"]["n"]) for entry in result.trajectory] == [(0, 1), (1, 1), (2, 2), (3, 3)]
    # Its own max_depth counts from where it was called: its tool ran at its first two levels and not below.
    assert ran == [1, 2] and result.termination_reason == "max_depth"
    # It goes as deep as it would alone, whatever the depth of the module that called it: a module
    # behaves the same wherever it is used. What bounds them both is the run's requests and tokens.
    result = run(outer_depth=1, inner_depth=2)
    assert ran == [1, 2] and [entry["depth"] for entry in result.trajectory] == [0, 1, 2, 3]

    # The turns its loop may take are its own, too.
    inner = RDM(Tick, tools=[tick], max_iters=2)
    outer = RDM(Begin, tools=[dspy.Tool(inner, name="count")], max_iters=20)
    client = Client(lambda inputs, name, question: "tick" if name == "next" else True)
    inner.set_lm(client)
    outer.set_lm(client)
    result = outer(counter={"n": 0})
    assert [(entry["depth"], entry["iteration"], entry.get("skipped")) for entry in result.trajectory] == [
        (0, 0, None), (1, 0, None), (1, 1, None), (1, 2, "max_iters"),
    ]


def test_a_module_called_inside_a_tool_keeps_its_own_limits_and_the_runs():
    def run(inner_calls, outer_calls, inner_tokens=None, outer_tokens=None, tokens=0):
        inner = RDM(Tick, tools=[tick], max_predictor_calls=inner_calls, max_tokens=inner_tokens)
        outer = RDM(Begin, tools=[dspy.Tool(inner, name="count")], max_predictor_calls=outer_calls, max_tokens=outer_tokens)
        client = Client(lambda inputs, name, question: "tick" if name == "next" else True, tokens=tokens)
        inner.set_lm(client)
        outer.set_lm(client)
        counter = {"n": 0}
        return outer(counter=counter), counter["n"]

    # Alone, the inner module stops at its three requests. Called inside a tool it joins the run,
    # and its own limit still holds for what it does there.
    result, ticks = run(inner_calls=3, outer_calls=100)
    assert ticks == 3 and result.termination_reason == "max_predictor_calls"
    assert [entry.get("skipped") for entry in result.trajectory] == [None, None, None, None, "max_predictor_calls"]
    # The run's limit holds for the inner module too: here it is the smaller.
    result, ticks = run(inner_calls=100, outer_calls=4)
    assert ticks == 3 and result.termination_reason == "max_predictor_calls"
    # The same for tokens: each module's limit is on what is used under it.
    result, ticks = run(inner_calls=100, outer_calls=100, inner_tokens=250, tokens=100)
    assert ticks == 3 and result.termination_reason == "max_tokens"
    result, ticks = run(inner_calls=100, outer_calls=100, outer_tokens=250, tokens=100)
    assert ticks == 2 and result.termination_reason == "max_tokens"


def test_a_pointer_can_be_given_no_finish():
    class Play(dspy.Signature):
        """Play."""

        board: dict = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="Which move?")

    def left(board):
        board["x"] -= 1

    def right(board):
        board["x"] += 1

    # Without `finish` there is nothing to stop at but a limit or a repeat: a game is played to its end.
    client = Client(lambda inputs, name, question: "right")
    rdm = RDM(Play, tools=[left, right], finish=False, max_iters=5)
    rdm.set_lm(client)
    board = {"x": 0}
    result = rdm(board=board)
    assert all(list(call["questions"]["next"]["criteria"]) == ["left", "right"] for call in client.calls)
    assert board["x"] == 5 and result.termination_reason == "max_iters"
    # A tool named `finish` still ends the loop: it is a tool the caller chose to offer.
    rdm = RDM(Play, tools=[right, dspy.Tool(lambda: "over", name="finish", desc="The game is over.")], finish=False)
    rdm.set_lm(Client(lambda inputs, name, question: "finish" if inputs["board"]["x"] >= 2 else "right"))
    assert rdm(board={"x": 0}).observations["next"][-1] == ["finish", "over"]
    # With one tool and no `finish` there is nothing to decide: it runs without a request, turn after turn.
    client = Client(lambda inputs, name, question: "right")
    rdm = RDM(Play, tools=[right], finish=False, max_iters=3)
    rdm.set_lm(client)
    board = {"x": 0}
    assert rdm(board=board).termination_reason == "max_iters" and board["x"] == 3 and client.calls == []


def test_requests_turns_and_tokens_are_bounded_and_what_was_cut_is_recorded():
    holder = {}

    def again(n):
        return [holder["rdm"](n=2 * n), holder["rdm"](n=2 * n + 1)]

    # The number of requests answered is bounded across the whole run. A call that was not
    # answered comes back to the tool with no answers and the limit that stopped it.
    rdm = holder["rdm"] = RDM(Count, tools=[again], max_depth=10, max_predictor_calls=5)
    client = Client(lambda inputs, name, question: True)
    rdm.set_lm(client)
    result = rdm(n=1)
    cut = requests(result, skipped="max_predictor_calls")
    assert len(client.calls) == 5 and len(cut) > 0 and all("outputs" not in entry for entry in cut)
    refused = [sub for entry in result.trajectory for sub in entry.get("observations", {}).get("again", []) if sub.again is None]
    assert refused and {sub.termination_reason for sub in refused} == {"max_predictor_calls"}

    # So are the tokens the requests use, as the client reports them.
    rdm = holder["rdm"] = RDM(Count, tools=[again], max_depth=10, max_tokens=350)
    client = Client(lambda inputs, name, question: True, tokens=100)
    rdm.set_lm(client)
    result = rdm(n=1)
    assert len(client.calls) == 4 and sum(entry.get("tokens", 0) for entry in result.trajectory) == 400
    assert len(requests(result, skipped="max_tokens")) > 0

    # Turns are the times a pointer's question is put in a row.
    class Steps(dspy.Signature):
        """Take steps."""

        n: int = dspy.InputField()
        observation: str = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="What next?")

    steps = iter(range(100))
    rdm = RDM(Steps, tools=[dspy.Tool(lambda: next(steps), name="step")], max_iters=3)
    client = Client(lambda inputs, name, question: "step")
    rdm.set_lm(client)
    result = rdm(n=0)
    assert [(entry["depth"], entry["iteration"], entry.get("skipped")) for entry in result.trajectory] == [
        (0, 0, None), (0, 1, None), (0, 2, None), (0, 3, "max_iters"),
    ]
    assert len(client.calls) == 3 and result.termination_reason == "max_iters"


def test_a_tool_that_fails_is_an_observation_and_not_the_end_of_the_run():
    def refund(ticket):
        raise TimeoutError("the ledger did not answer")

    def escalate(ticket):
        return "paged"

    rdm = RDM(Triage, tools=[refund, escalate])
    rdm.set_lm(Client(lambda inputs, name, question: True if question["type"] == "noul" else 0))
    result = rdm(ticket="t")
    assert result.observations["refund"].startswith("Execution error in refund:")
    assert "the ledger did not answer" in result.observations["refund"] and result.observations["escalate"] == "paged"


def test_tools_are_called_as_dspy_tools_so_callbacks_fire_and_async_tools_run():
    from dspy.utils.callback import BaseCallback

    class Seen(BaseCallback):
        def __init__(self):
            self.started = []

        def on_tool_start(self, call_id, instance, inputs):
            self.started.append((instance.name, inputs))

    async def refund(ticket):
        """Start a refund, somewhere that takes a while."""
        return f"refund started for {ticket!r}"

    def escalate(ticket, call: RDM.Call):
        return f"paged at depth {call.depth}"

    seen = Seen()
    rdm = RDM(Triage, tools=[refund, escalate])
    rdm.set_lm(Client(lambda inputs, name, question: True if question["type"] == "noul" else 0))
    # A tool is called as `dspy.Tool` is called, so whatever traces tools sees an RDM's.
    with dspy.context(callbacks=[seen], allow_tool_async_sync_conversion=True):
        result = rdm(ticket="t")
    assert sorted((name, sorted(inputs["kwargs"])) for name, inputs in seen.started) == [("escalate", ["call", "ticket"]), ("refund", ["ticket"])]
    assert result.observations == {"refund": "refund started for 't'", "escalate": "paged at depth 0"}
    # An async tool is run as DSPy runs one: when called with `acall`, or with the conversion allowed.
    assert asyncio.run(rdm.acall(ticket="t")).observations["refund"] == "refund started for 't'"
    assert "allow_tool_async_sync_conversion" in rdm(ticket="t").observations["refund"]


# Thresholds and ReAnchor


def test_thresholds_are_the_predictors_per_field_parameters():
    def refund(ticket):
        return "refund started"

    client = Client(lambda inputs, name, question: {"refund": 0.6, "escalate": 0.2, "frustration": 0}[name])
    rdm = RDM(Triage, tools=[refund])
    rdm.set_lm(client)
    assert rdm(ticket="t").observations == {"refund": "refund started"}
    # The model said 0.6. DSPy applies the threshold, and the tool is no longer called.
    rdm.predict.fields["refund"] = {"threshold": 0.7}
    result = rdm(ticket="t")
    assert (result.refund.probability, result.refund.value, result.observations) == (0.6, False, {})
    assert "threshold" not in json.dumps(client.calls)


def test_a_threshold_belongs_to_an_output_name_whichever_request_asks_it():
    class Start(dspy.Signature):
        """Decide whether to look."""

        question: str = dspy.InputField()
        look: Noul = dspy.OutputField(desc="Is it worth looking?")

    class Check(dspy.Signature):
        """Check a note."""

        note: str = dspy.InputField()
        look: Noul = dspy.OutputField(desc="Is it worth looking further?")
        useful: Noul = dspy.OutputField(desc="Is the note useful?")

    def look(call: RDM.Call):
        return call.rdm(signature=Check, note=f"note {len(call.trajectory)}")

    rdm = RDM(Start, tools=[look], max_depth=4)
    client = Client(lambda inputs, name, question: 0.6)
    rdm.set_lm(client)
    assert len(rdm(question="q").trajectory) == 5
    # The threshold is set once, on the predictor, and applies wherever `look` is asked: in the
    # first request and in requests whose questions were built at run time.
    rdm.predict.fields["look"] = {"threshold": 0.7}
    assert len(rdm(question="q").trajectory) == 1
    # One can be set for an output that only a later request asks, and it applies there.
    rdm.predict.fields = {"useful": {"threshold": 0.7}}
    result = rdm(question="q")
    assert result.trajectory[1]["outputs"]["useful"]["value"] is False and result.trajectory[1]["outputs"]["look"]["value"] is True


def test_a_predictor_given_to_the_module_answers_the_requests_that_ask_its_questions(tmp_path):
    def look(question, archive, call: RDM.Call):
        return [call.rdm(signature=Judge, record=record, question=question).answers.value for record in archive.values()]

    # `Judge` is asked only in sub-calls. Held by the module as a predictor, it is a parameter of the
    # program like the module's own: named, given the module's client, saved, and fitted by ReAnchor.
    rdm = RDM(Find, tools=[look])
    rdm.judge = dspy.Predict(Judge)
    assert [name for name, _ in rdm.named_predictors()] == ["predict", "judge"]
    rdm.set_lm(Client(lambda inputs, name, question: True if name == "look" else 0.6))
    archive = {"a": "The hangar opens at dawn.", "b": "The hangar door sticks in frost."}
    assert rdm(archive=archive, question="q").observations == {"look": [True, True]}
    rdm.judge.fields["answers"] = {"threshold": 0.7}
    assert rdm(archive=archive, question="q").observations == {"look": [False, False]}
    saved = RDM(Find, tools=[look])
    saved.judge = dspy.Predict(Judge)
    saved.judge.fields["answers"] = {"threshold": 0.7}
    saved.save(tmp_path / "rdm.json")
    restored = RDM(Find, tools=[look])
    restored.judge = dspy.Predict(Judge)
    restored.load(tmp_path / "rdm.json")
    assert restored.judge.fields == {"answers": {"threshold": 0.7}}

    # The model says 0.6 for the record that answers and 0.55 for the one that does not.
    def answer(inputs, name, question):
        return True if name == "look" else 0.6 if "dawn" in inputs["record"] else 0.55

    rdm = RDM(Find, tools=[look])
    rdm.judge = dspy.Predict(Judge)
    rdm.set_lm(Client(answer))
    trainset = [dspy.Example(archive=archive, question=f"q{n}").with_inputs("archive", "question") for n in range(6)]
    optimizer = ReAnchor(lambda example, prediction, trace=None: float(prediction.observations.get("look") == [True, False]))
    tuned = optimizer.compile(rdm, trainset=trainset)
    assert (optimizer.report["train_score_before"], optimizer.report["train_score"]) == (0.0, 1.0)
    assert 0.55 < tuned.judge.fields["answers"]["threshold"] < 0.6


def test_parameters_fitted_to_one_answer_space_are_not_applied_to_another():
    # `descend` is asked at every node of the taxonomy, each time over other options. Weights
    # set for the root's options apply to the root, and the requests below are answered without them.
    rdm = RDM(ask([]), tools=[descend])
    rdm.set_lm(Client(pick))
    rdm.predict.fields["descend"] = {"weights": {"Animals & Pet Supplies": 1.0, "Furniture": 0.5}}
    result = rdm(listing=LISTING, path=[])
    assert result.observations["descend"] == ["Animals & Pet Supplies", "Cat Supplies", "Cat Window Perches"]


def test_reanchor_fits_when_a_tool_is_called_against_the_whole_program():
    class Dig(dspy.Signature):
        """Dig until the layer is solid."""

        layer: int = dspy.InputField()
        solid_at: int = dspy.InputField()
        dig: Noul = dspy.OutputField(desc="Is the layer still loose?")

    def dig(layer, solid_at, call: RDM.Call):
        # `call.rdm` is the module that is running: ReAnchor runs a copy, and its own thresholds apply below.
        return call.rdm(layer=layer + 1, solid_at=solid_at)

    # The model is never sure a layer is solid: it says 0.6 where it should say no.
    client = Client(lambda inputs, name, question: 0.9 if inputs["layer"] < inputs["solid_at"] else 0.6)
    rdm = RDM(Dig, tools=[dig], max_depth=8)
    rdm.set_lm(client)
    trainset = [dspy.Example(layer=0, solid_at=n).with_inputs("layer", "solid_at") for n in range(1, 6)]

    def metric(example, prediction, trace=None):
        return float(len(requests(prediction, skipped=None)) == example.solid_at + 1)

    optimizer = ReAnchor(metric)
    tuned = optimizer.compile(rdm, trainset=trainset)
    # At 0.5 every run digs as deep as it may. ReAnchor finds the cut between 0.6 and 0.9.
    assert (optimizer.report["train_score_before"], optimizer.report["train_score"]) == (0.0, 1.0)
    assert 0.6 < tuned.predict.fields["dig"]["threshold"] < 0.9 and rdm.predict.fields == {}
    assert [(row["predictor"], row["field"]) for row in optimizer.report["fitted"] if "skipped" not in row] == [("predict", "dig")]


# Tools that are modules, and other backends


def test_a_tool_can_be_any_module_a_generative_predict_or_another_rdm():
    class Route(dspy.Signature):
        """Route a request to a smart home assistant."""

        request: str = dspy.InputField()
        chat: Noul = dspy.OutputField(desc="Is the request a question to answer in conversation?")
        devices: Noul = dspy.OutputField(desc="Is the request a command to a device?")

    class Device(dspy.Signature):
        """Carry out a command to a device."""

        request: str = dspy.InputField()
        lock: Noul = dspy.OutputField(desc="Should the door be locked?")

    locked = []
    chat = dspy.Predict("request -> reply")
    chat.set_lm(DummyLM([{"reply": "It is 21 degrees."}]))
    child = RDM(Device, tools=[dspy.Tool(lambda request: locked.append(request), name="lock")])
    child.set_lm(Client(lambda inputs, name, question: True))
    rdm = RDM(Route, tools=[dspy.Tool(chat, name="chat"), dspy.Tool(child, name="devices")])

    # A generative predict is called through its signature. What it wrote comes back as a result.
    rdm.set_lm(Client(lambda inputs, name, question: name == "chat"))
    result = rdm(request="How warm is it?")
    assert result.observations["chat"].reply == "It is 21 degrees."
    assert "How warm is it?" in json.dumps(chat.lm.history[0]["messages"])
    # Another RDM called from a tool is a sub-call like any other: it joins the run, one level deeper.
    rdm.set_lm(Client(lambda inputs, name, question: name == "devices"))
    result = rdm(request="Lock up.")
    assert locked == ["Lock up."] and result.observations["devices"].lock.value is True
    assert [(entry["depth"], list(entry["outputs"])) for entry in result.trajectory] == [(0, ["chat", "devices"]), (1, ["lock"])]
    # Setting the client leaves a tool's own models alone, as with any DSPy tool.
    assert [name for name, _ in rdm.named_predictors()] == ["predict"] and chat.lm is not rdm.predict.lm


@pytest.mark.parametrize("adapter", [dspy.ChatAdapter(), dspy.JSONAdapter()])
def test_a_generative_lm_can_answer_the_same_requests(adapter):
    class Recording(Client):
        answers = ()

        def __call__(self, state, questions):
            answers = super().__call__(state, questions)
            self.answers = (*self.answers, answers)
            return answers

    decided = RDM(ask([]), tools=[descend])
    client = Recording(pick)
    decided.set_lm(client)
    expected = decided(listing=LISTING, path=[])
    # The same module on a generative LM: it is asked for the same probabilities, as structured
    # output, and DSPy derives the same decisions from them.
    lm = DummyLM(list(client.answers), adapter=adapter)
    generated = RDM(ask([]), tools=[descend])
    generated.set_lm(lm)
    with dspy.context(adapter=adapter):
        result = generated(listing=LISTING, path=[])
    assert [entry["outputs"] for entry in result.trajectory] == [entry["outputs"] for entry in expected.trajectory]
    assert len(lm.history) == 3
    assert result.observations["descend"][-1] == "Cat Window Perches"


@pytest.mark.asyncio
async def test_it_can_be_awaited_copied_saved_and_loaded(tmp_path):
    rdm = RDM(ask([]), tools=[descend])
    client = Client(pick)
    rdm.set_lm(client)
    result = await rdm.acall(listing="Wall shelves for the furniture and shelving aisle.", path=[])
    assert [entry["outputs"]["descend"]["value"] for entry in result.trajectory] == ["Furniture", "Shelving", "Wall Shelves"]

    rdm = RDM(Triage)
    rdm.predict.fields["refund"] = {"threshold": 0.8}
    copy = rdm.deepcopy()
    assert copy.predict.fields == {"refund": {"threshold": 0.8}} and copy.predict is not rdm.predict
    rdm.save(tmp_path / "rdm.json")
    restored = RDM(Triage)
    restored.load(tmp_path / "rdm.json")
    assert restored.predict.fields == {"refund": {"threshold": 0.8}}
