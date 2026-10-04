"""Tests for Fit, which cuts a request for the RDM module down to one that a single request holds.

Test organization:
- An input too long for one request
- A Choice with too many options
- Options that are the keys of an input
- Limits, and sub-calls
"""

import json
from typing import Any

import pytest

import dspy
from dspy.experimental import RDM, Choice, Noul
from tests.predict.test_rdm import Client, requests

Fit = RDM.Fit


class Ask(dspy.Signature):
    """Decide whether to look in the archive."""

    question: str = dspy.InputField()
    look: Noul = dspy.OutputField(desc="Is `inputs.question` one that an archive of records could answer?")


# An input too long for one request




class Covered(dspy.Signature):
    """Decide whether a handbook covers a question."""

    handbook: str = dspy.InputField()
    question: str = dspy.InputField()
    covered: Noul = dspy.OutputField(desc="Does `inputs.handbook` state the answer to `inputs.question`?")


RULES = [f"Rule {n}: staff badge {n} opens door {n} on weekdays.\n" for n in range(2000)]
RULES[1203] = "Rule 1203: the nightly hotel limit in London is 180 pounds.\n"
HANDBOOK = "".join(RULES)
HOTEL = "What is the nightly hotel limit in London?"


def test_an_input_too_long_for_one_request_is_narrowed_to_the_parts_the_questions_need():
    client = Client(lambda inputs, name, question: "hotel limit" in inputs["handbook"])
    rdm = RDM(Covered, fit=Fit(max_chars=8000))
    rdm.set_lm(client)
    result = rdm(handbook=HANDBOOK, question=HOTEL)
    assert len(HANDBOOK) > 90_000
    # The handbook was taken in parts. Each part is a request of its own: the part, the inputs that
    # fit, the questions the inputs are for, and one question, whether the part is needed.
    parts = requests(result, depth=1)
    assert len(parts) == len(client.calls) - 1 > 90 and {entry["parent"] for entry in parts} == {0}
    assert "".join(entry["inputs"]["handbook"] for entry in parts) == HANDBOOK
    assert all(entry["inputs"]["question"] == HOTEL and list(entry["outputs"]) == ["needed"] for entry in parts)
    assert parts[0]["inputs"]["questions"] == {"covered": "Does `inputs.handbook` state the answer to `inputs.question`?"}
    assert "`inputs.handbook` is one part of a longer input" in client.calls[0]["questions"]["needed"]["instructions"]
    # The signature's own question was then asked once, of the part that was needed and, in the
    # room left, the likeliest of the others.
    last = client.calls[-1]
    assert list(last["questions"]) == ["covered"] and last["state"]["inputs"]["question"] == HOTEL
    assert RULES[1203] in last["state"]["inputs"]["handbook"] and 1000 < len(last["state"]["inputs"]["handbook"]) < 8000
    assert result.trajectory[0]["inputs"] == last["state"]["inputs"]
    # No request held more than a request may, and the answer is the one a request with room for
    # the whole handbook gives.
    assert max(len(json.dumps(call["state"]["inputs"])) for call in client.calls) < 8000
    with dspy.context(lm=client):
        whole = dspy.Predict(Covered)(handbook=HANDBOOK, question=HOTEL)
    assert (result.covered.value, result.termination_reason) == (whole.covered.value, "final") == (True, "final")
    # A question the handbook does not answer is put to the likeliest part, and answered no.
    client.answer = lambda inputs, name, question: 0.2 if "Rule 7:" in inputs["handbook"] else 0.1
    result = rdm(handbook=HANDBOOK, question="Who founded the company?")
    assert "Rule 7:" in result.trajectory[0]["inputs"]["handbook"] and result.covered.value is False


def test_parts_keep_their_type_and_order_and_a_text_shows_where_parts_were_left_out():
    class Mentions(dspy.Signature):
        """Decide whether the notes mention a topic."""

        notes: Any = dspy.InputField()
        topic: str = dspy.InputField()
        mentioned: Noul = dspy.OutputField(desc="Do `inputs.notes` mention `inputs.topic`?")

    def narrowed(notes, max_chars=400):
        rdm = RDM(Mentions, fit=Fit(max_chars=max_chars))
        rdm.fit.needed.fields["needed"] = {"threshold": 0.5}  # only the parts that are needed, so what is kept is exact
        rdm.set_lm(Client(lambda inputs, name, question: "kiln" in json.dumps(inputs["notes"])))
        return rdm(notes=notes, topic="the kiln").trajectory[0]["inputs"]["notes"]

    # A text is split between lines, and what is kept is joined with a mark where parts are missing.
    lines = [f"Line {n} is about the weather.\n" for n in range(60)]
    lines[5], lines[40] = "Line 5: the kiln is hot.\n", "Line 40: the kiln is cold.\n"
    assert narrowed("".join(lines)) == lines[5] + "\n[...]\n" + lines[40]
    # A list is split between items and stays a list, and a dict between entries and stays a dict.
    items = [line.strip() for line in lines]
    assert narrowed(items) == [items[5], items[40]]
    assert narrowed({f"n{n}": item for n, item in enumerate(items)}) == {"n5": items[5], "n40": items[40]}
    # A line or an item longer than a part is split wherever it has to be.
    assert "kiln" in narrowed("x" * 3000 + " kiln " + "y" * 3000) and "kiln" in narrowed(["x" * 3000 + " kiln " + "y" * 3000])[0]
    # What cannot be taken in parts is refused.
    with pytest.raises(ValueError, match="An input of type int is too long for one request"):
        narrowed(10**500)

    # A sub-call is narrowed like any call, and so is a call at max_depth: every request has to fit.
    def look(question, call: RDM.Call):
        return call.rdm(signature=Covered, handbook=call.state["handbook"], question=question).covered.value

    rdm = RDM(Ask, tools=[look], fit=Fit(max_chars=8000), max_depth=1)
    rdm.set_lm(Client(lambda inputs, name, question: name == "look" or "hotel limit" in inputs["handbook"]))
    result = rdm(question=HOTEL, state={"handbook": HANDBOOK})
    assert result.observations == {"look": True} and len(requests(result, depth=2)) > 90
    assert RULES[1203] in result.trajectory[1]["inputs"]["handbook"] and len(result.trajectory[1]["inputs"]["handbook"]) < 8000


def test_the_likeliest_parts_fill_the_room_unless_a_threshold_is_set(tmp_path):
    notes = "".join(f"Line {n} is about the weather.\n" for n in range(60))

    def answer(inputs, name, question):
        return {"Line 3 ": 0.35, "Line 30 ": 0.3}.get(next((mark for mark in ("Line 3 ", "Line 30 ") if mark in inputs["handbook"]), ""), 0.02)

    # A model may rate what is needed below one half. By default the likeliest parts are kept, as
    # many as there is room for, whatever their ratings: a fixed cut means something different on each model.
    rdm = RDM(Covered, fit=Fit(max_chars=400))
    rdm.set_lm(Client(answer))
    kept = rdm(handbook=notes, question="q").trajectory[0]["inputs"]["handbook"]
    assert "Line 3 " in kept and "Line 30 " in kept and len(kept) > 200
    # The question put to each part is a predictor of the program, `fit.needed`. Given a threshold,
    # it keeps only the parts it says are needed. The threshold is saved and loaded with the module.
    rdm = RDM(Covered, fit=Fit(max_chars=400))
    assert [name for name, _ in rdm.named_predictors()] == ["predict", "fit.needed"]
    rdm.fit.needed.fields["needed"] = {"threshold": 0.33}
    rdm.save(tmp_path / "rdm.json")
    rdm.set_lm(Client(answer))
    kept = rdm(handbook=notes, question="q").trajectory[0]["inputs"]["handbook"]
    assert "Line 3 " in kept and "Line 30 " not in kept and len(kept) < 100
    restored = RDM(Covered, fit=Fit(max_chars=400))
    restored.load(tmp_path / "rdm.json")
    assert restored.fit.needed.fields == {"needed": {"threshold": 0.33}}


def test_reanchor_fits_the_threshold_of_needed_against_the_whole_program():
    # The model rates the part that answers at 0.9 and a misleading part at 0.4. With no threshold
    # both fit in the room, and the misleading one spoils the answer.
    def notes(position, stated=True):
        lines = [f"Note {n}: the canteen serves soup on day {n} of every month, as it always has.\n" for n in range(40)]
        if stated:
            lines[position] = "The nightly hotel limit in London is 180 pounds, breakfast included, since January.\n"
        lines[(position + 17) % 40] = "Decoy: a hotel limit of 500 pounds was proposed once and never adopted by the board.\n"
        return "".join(lines)

    def answer(inputs, name, question):
        text = inputs["handbook"]
        if name == "needed":
            return 0.9 if "nightly hotel limit" in text else 0.4 if "Decoy" in text else 0.05
        return "nightly hotel limit" in text and "Decoy" not in text

    rdm = RDM(Covered, fit=Fit(max_chars=800))
    rdm.set_lm(Client(answer))
    # Half the handbooks state the limit and half do not, so no threshold on the answer itself can win.
    trainset = [
        dspy.Example(handbook=notes(position, stated), question=HOTEL, stated=stated).with_inputs("handbook", "question")
        for position in range(3, 9) for stated in (True, False)
    ]
    assert all("Decoy" in rdm(**example.inputs()).trajectory[0]["inputs"]["handbook"] for example in trainset[:2])

    optimizer = dspy.experimental.ReAnchor(lambda example, prediction, trace=None: float(prediction.covered.value == example.stated))
    tuned = optimizer.compile(rdm, trainset=trainset)
    assert (optimizer.report["train_score_before"], optimizer.report["train_score"]) == (0.5, 1.0)
    assert 0.4 < tuned.fit.needed.fields["needed"]["threshold"] < 0.9 and rdm.fit.needed.fields == {}
    assert ("fit.needed", "needed") in [(row["predictor"], row["field"]) for row in optimizer.report["fitted"] if "skipped" not in row]


def test_options_that_are_not_candidates_are_offered_to_every_group():
    # `finish` beside three hundred tools: every group of the shortlist offers it, as `None` is offered.
    class Act(dspy.Signature):
        """Act."""

        task: str = dspy.InputField()
        next: dspy.Tool = dspy.OutputField(desc="What next?")

    tools = [dspy.Tool(lambda: "done", name=f"tool {n}", desc="") for n in range(300)]
    client = Client(lambda inputs, name, question: "finish" if "finish" in question["criteria"] and len(question["criteria"]) < 10 else next(iter(question["criteria"])))
    rdm = RDM(Act, tools=tools, fit=Fit(max_options=10), always=["tool 299"])
    rdm.set_lm(client)
    result = rdm(task="t")
    assert all("finish" in question["criteria"] for call in client.calls for question in call["questions"].values())
    assert result.observations["next"][-1] == ["finish", None]
    # So are the names the module always offers: a tool that must stay in reach whatever is cut.
    assert all("tool 299" in question["criteria"] for call in client.calls for question in call["questions"].values())

    # A Choice whose options are a dict's keys and "none of these" stays keyed: each group is
    # asked about its own entries, and the final request shows the entries of its options.
    rows = {f"f{n:03d}": {"status": "stopped" if n == 250 else "working"} for n in range(300)}
    pick = Choice[(*((key, "") for key in rows), ("none", "None has stopped."))]
    signature = dspy.Signature({"furnaces": (dict, dspy.InputField()), "pick": (pick, dspy.OutputField(desc="Which furnace has stopped?"))})
    client = Client(lambda inputs, name, question: next((key for key in question["criteria"] if inputs["furnaces"].get(key, {}).get("status") == "stopped"), "none"))
    rdm = RDM(signature, fit=Fit(max_chars=4000, max_options=20))
    rdm.set_lm(client)
    assert rdm(furnaces=rows).pick.value == "f250"
    for call in client.calls:
        offered = set(call["questions"]["pick"]["criteria"]) - {"none"}
        assert offered == set(call["state"]["inputs"]["furnaces"]) and "none" in call["questions"]["pick"]["criteria"]


def test_a_choice_with_more_options_than_a_request_takes_is_cut_to_a_shortlist():
    def pick(count):
        Item = Choice[tuple((f"item-{n}", "") for n in range(count))]  # noqa: N806
        return dspy.Signature(
            {
                "request": (str, dspy.InputField()),
                "item": (Item, dspy.OutputField(desc="Which item does `inputs.request` name?")),
                "urgent": (Noul, dspy.OutputField(desc="Is `inputs.request` urgent?")),
            },
            "Read an order.",
        )

    def answer(inputs, name, question):
        return True if name == "urgent" else "item-777" if "item-777" in question["criteria"] else next(iter(question["criteria"]))

    # A thousand options at 255 to a request: the question is put to four groups, alone, and each
    # group's likeliest 63 go on, which is as many as leave room for all four groups' in one request.
    # The groups are asked about the same inputs, so they are four questions of one request: the
    # inputs are sent once.
    client = Client(answer)
    assert Fit().max_options == 240  # the default leaves room under the 255 a request takes
    # Every size is an argument, since each suits one model: here a model that takes one question to a request.
    # What Fit asks is for the answers only: a tool named after the question is run by the request itself, once.
    one, sent = Client(answer), []
    rdm = RDM(pick(1000), tools=[dspy.Tool(lambda item: sent.append(item), name="item")], fit=Fit(max_options=255, max_questions=1, keep=1))
    rdm.set_lm(one)
    assert rdm(request="Send item-777 today.").item.value == "item-777" and sent == ["item-777"]
    assert [[len(question["criteria"]) for question in call["questions"].values()] for call in one.calls[:4]] == [[255], [255], [255], [235]]
    with pytest.raises(ValueError, match="must each be at least 1"):
        Fit(parts=0)
    rdm = RDM(pick(1000), fit=Fit(max_options=255))
    rdm.set_lm(client)
    result = rdm(request="Send item-777 today.")
    assert len(requests(result, depth=1)) == 1
    first = client.calls[0]
    assert list(first["questions"]) == ["item_0", "item_1", "item_2", "item_3"] and first["state"]["inputs"] == {"request": "Send item-777 today."}
    assert [len(question["criteria"]) for question in first["questions"].values()] == [255, 255, 255, 235]
    assert len({question["instructions"] for question in first["questions"].values()}) == 1
    # Then the request itself is asked once, every question together, over the shortlist.
    assert list(client.calls[-1]["questions"]) == ["item", "urgent"] and "item-777" in client.calls[-1]["questions"]["item"]["criteria"]
    assert len(client.calls[-1]["questions"]["item"]["criteria"]) <= 255
    assert (result.item.value, result.urgent.value, len(client.calls)) == ("item-777", True, 2)

    # With far more options than fit, it takes more rounds. A group sends on two where it can, so
    # that two good options in one group both reach the next round.
    client = Client(answer)
    rdm = RDM(pick(5000), fit=Fit(max_options=10), max_predictor_calls=1000)
    rdm.set_lm(client)
    result = rdm(request="Send item-777 today.")
    asked = [len(entry["outputs"]) for entry in requests(result, depth=1)]
    assert sum(asked) == 500 + 100 + 20 + 4 and len(asked) == 32 + 7 + 2 + 1 and result.item.value == "item-777"
    assert max(len(question["criteria"]) for call in client.calls for question in call["questions"].values() if question["type"] == "choice") == 10


def test_options_too_long_for_a_request_are_grouped_by_length_as_well_as_by_number():
    # Sixty options is few enough for one request, but each is described at length: together they
    # are several times what a request shows. So the groups are as many as fit by their length.
    described = tuple((f"n{n}", f"Category {n}: " + "a long path through a taxonomy, " * 30) for n in range(60))
    signature = dspy.Signature(
        {
            "listing": (str, dspy.InputField()),
            "category": (Choice[described], dspy.OutputField(desc="Which category does `inputs.listing` belong to?")),
        },
        "Classify a listing.",
    )
    client = Client(lambda inputs, name, question: "n41" if "n41" in question["criteria"] else next(iter(question["criteria"])))
    rdm = RDM(signature, fit=Fit(max_chars=10_000))
    rdm.set_lm(client)
    result = rdm(listing="A listing for category 41.")
    lengths = [len(json.dumps(call["questions"]["category"]["criteria"])) for call in client.calls]
    assert sum(len(description) for _, description in described) > 55_000 and max(lengths) < 11_000
    assert len(requests(result, depth=1)) >= 6 and result.category.value == "n41"
    # One option that is longer than a request by itself cannot be shortened, and is asked as it is.
    rdm = RDM(signature, fit=Fit(max_chars=500))
    rdm.set_lm(Client(lambda inputs, name, question: "n41" if "n41" in question["criteria"] else next(iter(question["criteria"]))))
    assert rdm(listing="A listing for category 41.").category.value == "n41"
    # Where a request has room for two options and no more, a group is two and one of them goes on.
    brief = tuple((f"n{n}", f"Category {n}: " + "a path, " * 35) for n in range(40))
    signature = signature.with_updated_fields("category", type_=Choice[brief])
    client = Client(lambda inputs, name, question: "n41" if "n41" in question["criteria"] else sorted(question["criteria"])[-1])
    rdm = RDM(signature, fit=Fit(max_chars=700))
    rdm.set_lm(client)
    result = rdm(listing="A listing for category 39.")
    assert max(len(question["criteria"]) for call in client.calls for question in call["questions"].values()) == 2
    assert result.category.value == "n9" and result.termination_reason == "final"


def service(machines, none=False):
    """Which furnace to service: one option per entry of `machines`, each option an entry's key."""
    options = tuple((name, "") for name in machines) + (((None, "No furnace has stopped."),) if none else ())
    return dspy.Signature(
        {
            "machines": (dict, dspy.InputField(desc="Every furnace by its id.")),
            "shift": (str, dspy.InputField()),
            "urgent": (Choice[options], dspy.OutputField(desc="Which furnace in `inputs.machines` has stopped?")),
            "any": (Noul, dspy.OutputField(desc="Has any furnace in `inputs.machines` stopped?")),
        },
        "Keep a factory fed.",
    )


def stopped(inputs, name, question):
    """Answers with the furnace that has stopped, when the request shows one, and otherwise with its first option."""
    down = [key for key, machine in inputs["machines"].items() if machine["status"] != "working"]
    if name == "any":
        return bool(down)
    return down[0] if down else next(iter(question["criteria"]))


def test_a_choice_whose_options_are_the_keys_of_an_input_is_cut_down_together_with_it():
    machines = {f"f{n:04d}": {"status": "working", "fuel_lasts_s": 100 + n} for n in range(600)}
    machines["f0417"]["status"] = "no_fuel"
    client = Client(stopped)
    rdm = RDM(service(machines), fit=Fit(max_options=40))
    rdm.set_lm(client)
    result = rdm(machines=machines, shift="night")
    # Every option points at an entry, so the two are taken together: a group is forty entries
    # with exactly their keys as its options, and it is asked the question itself. No request
    # asks whether a part is needed, so no furnace is left out without the question being put to it.
    groups = requests(result, depth=1)
    assert len(groups) == 15 and {entry["parent"] for entry in groups} == {0}
    assert all(list(call["questions"]) == ["urgent"] and call["state"]["inputs"]["shift"] == "night" for call in client.calls[:-1])
    assert all(set(call["state"]["inputs"]["machines"]) == set(call["questions"]["urgent"]["criteria"]) for call in client.calls)
    assert max(len(call["state"]["inputs"]["machines"]) for call in client.calls) == 40
    assert sorted(key for entry in groups for key in entry["inputs"]["machines"]) == sorted(machines)
    # Each group's likeliest two go on, thirty in all, and the request is asked once of those:
    # every question together, about the entries that are left.
    last = client.calls[-1]
    assert list(last["questions"]) == ["urgent", "any"] and len(last["state"]["inputs"]["machines"]) == 30
    assert (result.urgent.value, result.any.value, len(client.calls)) == ("f0417", True, 16)
    assert result.trajectory[0]["inputs"]["machines"]["f0417"] == machines["f0417"]

    # Choosing none of them is open to every group and is no entry. When nothing has stopped, it wins the request.
    working = {name: {**machine, "status": "working"} for name, machine in machines.items()}
    client = Client(lambda inputs, name, question: stopped(inputs, name, question) if any(m["status"] != "working" for m in inputs["machines"].values()) else {"urgent": "None", "any": False}[name])
    rdm = RDM(service(working, none=True), fit=Fit(max_options=40))
    rdm.set_lm(client)
    result = rdm(machines=working, shift="night")
    assert all("None" in call["questions"]["urgent"]["criteria"] and "None" not in call["state"]["inputs"]["machines"] for call in client.calls)
    assert max(len(call["questions"]["urgent"]["criteria"]) for call in client.calls) == 40 and result.urgent.value is None

    # Entries that are long are grouped by their length, in what the rest of the request leaves.
    long = {f"f{n:04d}": {"status": "working", "log": "smelting iron ore. " * 50} for n in range(60)}
    long["f0041"]["status"] = "no_ingredients"
    client = Client(stopped)
    rdm = RDM(service(long), fit=Fit(max_chars=10_000))
    rdm.set_lm(client)
    assert rdm(machines=long, shift="night").urgent.value == "f0041"
    assert max(len(json.dumps(call["state"]["inputs"])) for call in client.calls) < 10_500 and len(client.calls) > 6

    # A dict whose keys are not the options is an input like any other, and is narrowed by what is needed.
    client = Client(lambda inputs, name, question: any(m["status"] != "working" for m in inputs["machines"].values()) if name != "urgent" else "yes")
    signature = service(machines).with_updated_fields("urgent", type_=Choice[("yes", ""), ("no", "")])
    rdm = RDM(signature, fit=Fit(max_chars=8000))
    rdm.set_lm(client)
    result = rdm(machines=machines, shift="night")
    assert {list(entry["outputs"]) == ["needed"] for entry in requests(result, depth=1)} == {True}
    assert list(result.trajectory[0]["inputs"]["machines"]) == [key for key in machines if key in result.trajectory[0]["inputs"]["machines"]]
    assert "f0417" in result.trajectory[0]["inputs"]["machines"] and result.any.value is True


def test_cutting_a_request_down_counts_against_the_limits_and_in_order():
    rdm = RDM(Covered, fit=Fit(max_chars=8000), max_predictor_calls=10)
    client = Client(lambda inputs, name, question: "hotel limit" in inputs["handbook"])
    rdm.set_lm(client)
    result = rdm(handbook=HANDBOOK, question=HOTEL)
    # The request itself is counted first, so the requests that cut it down cannot leave it
    # unasked. With fewer requests left than parts, the parts are made longer so that as much of
    # the input as possible is asked about, and they are admitted in order, so the limit stops
    # the same ones every time: here all but the first nine.
    parts = requests(result, depth=1)
    assert len(parts) < 20 and ["skipped" in entry for entry in parts] == [False] * 9 + [True] * (len(parts) - 9)
    assert max(len(entry["inputs"]["handbook"]) for entry in parts) > 7000
    # The question was asked, of the part that was needed, and the call says what the limit left out.
    assert len(client.calls) == 10 and list(client.calls[-1]["questions"]) == ["covered"]
    assert (result.covered.value, result.termination_reason) == (True, "max_predictor_calls")
    # With requests to spare the parts are short, and all of the input is asked about.
    rdm = RDM(Covered, fit=Fit(max_chars=8000), max_predictor_calls=41)
    rdm.set_lm(Client(lambda inputs, name, question: "hotel limit" in inputs["handbook"]))
    result = rdm(handbook=HANDBOOK, question=HOTEL)
    parts = requests(result, depth=1)
    assert len(parts) <= 40 and not requests(result, skipped="max_predictor_calls") and "".join(entry["inputs"]["handbook"] for entry in parts) == HANDBOOK
    assert (result.covered.value, result.termination_reason) == (True, "final")

    rdm = RDM(Covered, fit=Fit(max_chars=8000), max_tokens=1000)
    client = Client(lambda inputs, name, question: "hotel limit" in inputs["handbook"], tokens=100)
    rdm.set_lm(client)
    result = rdm(handbook=HANDBOOK, question=HOTEL)
    assert 10 <= len(client.calls) < 20 and result.termination_reason == "max_tokens"
    assert len(requests(result, skipped="max_tokens")) == len(result.trajectory) - len(client.calls)


def test_a_tool_of_a_sub_call_is_given_that_calls_inputs_whole():
    class Check(dspy.Signature):
        """Check a handbook against a question."""

        handbook: str = dspy.InputField()
        question: str = dspy.InputField()
        check: Noul = dspy.OutputField(desc="Does `inputs.handbook` bear on `inputs.question`?")

    seen = {}

    def look(question, call: RDM.Call):
        return call.rdm(signature=Check, handbook=HANDBOOK, question=question).observations["check"]

    def check(handbook, call: RDM.Call):
        # Fit narrows what the model is shown. A tool is given the input whole, by name and on the
        # call, whichever call it is; the run's state holds only what was passed as `state=`.
        seen.update(given=len(handbook), whole=len(call.inputs["handbook"]), root="handbook" in call.state)
        return "checked"

    rdm = RDM(Ask, tools=[look, check], fit=Fit(max_chars=8000))
    rdm.set_lm(Client(lambda inputs, name, question: name in ("look", "check") or "hotel limit" in inputs["handbook"]))
    assert rdm(question=HOTEL).observations == {"look": "checked"}
    assert seen["given"] == seen["whole"] == len(HANDBOOK) and seen["root"] is False


def test_a_module_with_fit_can_be_copied_and_saved(tmp_path, recwarn):
    rdm = RDM(Ask, fit=Fit(max_chars=400))
    copy = rdm.deepcopy()
    assert copy.fit.max_chars == 400 and not [warning for warning in recwarn if "deep copy" in str(warning.message).lower()]
    rdm.save(tmp_path / "program", save_program=True)
    assert dspy.load(tmp_path / "program", allow_pickle=True).fit.max_chars == 400
