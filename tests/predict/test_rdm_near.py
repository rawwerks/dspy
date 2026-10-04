"""Tests for RDM.Near: labels for the names on a map, and the names nearest a request.

Test organization:
- Coordinates: a text's answers to named questions
- A map cut to what is nearest a request
- Dimensions a language model writes
- Thresholds and ReAnchor
- A program that is saved, loaded and copied
- With other helpers
"""

import pytest

import dspy
from dspy.experimental import RDM, Choice, Noul, ReAnchor, Score
from dspy.utils.dummies import DummyLM

Fit, Near = RDM.Fit, RDM.Near

# How likely each word makes each dimension. What a text says of a dimension is its likeliest word's.
WORDS = {
    "heating": {"cooler": 0.9, "warmer": 0.9, "degrees": 0.8, "stuffy": 0.6},
    "lights": {"light": 0.9, "lamp": 0.8, "dim": 0.5},
    "doors": {"door": 0.9, "lock": 0.8},
}


def picked(name, text):
    """What the stand-in chooses for a question that is not yes or no, by the words of the text."""
    if name == "urgency":  # a score: the index of its level
        return "2" if "today" in text else "1" if "week" in text else "0"
    if name == "room":
        return next((room for room in ("hall", "kitchen") if room in text), "None")
    # How a request uses a dimension: a filter where it says it must be that, a boost where it mentions it.
    dimension = name.removesuffix("_use")
    return "filter" if f"must be {dimension}" in text else "boost" if any(word in text for word in WORDS[dimension]) else "ignore"


class Client:
    """A stand-in System One client: a dimension is as likely as the words of the text make it, and a pointer
    takes the names of a script in turn, or with none the first tool offered, once."""

    supports_decision_requests = True
    cache = True
    model = "jev-test"

    def __init__(self, *script):
        self.script, self.calls, self.history, self.callbacks = list(script), [], [], []

    def __call__(self, state, questions):
        inputs = state["inputs"]
        self.calls.append({"inputs": inputs, "questions": {name: list(question.get("criteria", [])) for name, question in questions.items()}})
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"noul": max((p for word, p in WORDS[name].items() if word in inputs["request"].lower()), default=0.0)}
            elif not (name in ("urgency", "room") or name.endswith("_use")):
                chosen = self.script.pop(0) if self.script else "finish" if inputs.get("observation") else next(iter(question["criteria"]))
                answers[name] = {"probabilities": {label: float(label == chosen) for label in question["criteria"]}, "confidence": 1.0}
            else:
                labels = list(question["criteria"]) if question["type"] == "choice" else [str(level) for level in range(len(question["criteria"]))]
                answers[name] = {"probabilities": {label: float(label == picked(name, inputs["request"].lower())) for label in labels}, "confidence": 1.0}
        return answers

    def asked(self, name):
        return [call for call in self.calls if name in call["questions"]]


Urgency = Score["whenever", "this week", "today"]
Room = Choice[("hall", ""), ("kitchen", ""), (None, "No room is named.")]


class About(dspy.Signature):
    """Say what a text is about."""

    request: str = dspy.InputField()
    heating: Noul = dspy.OutputField(desc="Is `inputs.request` about how warm it is?")
    lights: Noul = dspy.OutputField(desc="Is `inputs.request` about a light?")


class Blank(dspy.Signature):
    """Say what a text is about."""

    request: str = dspy.InputField()


class Operate(dspy.Signature):
    """Operate a house, one step at a time."""

    request: str = dspy.InputField()
    observation: str = dspy.InputField()
    next: dspy.Tool = dspy.OutputField(desc="What should happen next for `inputs.request`?")


def steps():
    """A map of six lights and two heating steps, each a tool that says its own name."""
    described = [(f"{room} light", f"Turn the {room} light on.") for room in ("hall", "kitchen", "porch", "attic", "cellar", "study")]
    described += [("cooler", "Make it one degree cooler."), ("warmer", "Make it one degree warmer.")]
    return [dspy.Tool(lambda name=name: f"did {name}", name=name, desc=desc) for name, desc in described]


def house(near, client=None, **kwargs):
    rdm = RDM(Operate, tools=steps(), fit=near, **kwargs)
    rdm.set_lm(client or Client())
    return rdm


def offered(client):
    """What each request that pointed was offered."""
    return [call["questions"]["next"] for call in client.asked("next")]


# Coordinates: a text's answers to named questions


def test_a_text_has_a_coordinate_for_each_dimension_and_is_asked_once():
    near, client = Near(About), Client()
    near.set_lm(client)
    # Every number has a name: how likely a yes is, as the model answered.
    assert near.embed(["Make it cooler.", "The lamp in the hall."]) == [{"heating": 0.9, "lights": 0.0}, {"heating": 0.0, "lights": 0.8}]
    # A text is asked once, all its dimensions in one request, and what it answered is kept.
    assert near.embed(["The lamp in the hall.", "Nothing of either."]) == [{"heating": 0.0, "lights": 0.8}, {"heating": 0.0, "lights": 0.0}]
    assert [call["inputs"]["request"] for call in client.calls].count("The lamp in the hall.") == 1 and len(client.calls) == 3
    assert near.answers["Make it cooler."] == {"heating": 0.9, "lights": 0.0}
    # Things are compared by what their coordinates share: a search by named dimensions, with no map.
    cooler, lamp, both = near.embed(["Two degrees cooler.", "A lamp.", "A lamp, and warmer."])
    assert (near.score(cooler, cooler), near.score(cooler, lamp), near.score(cooler, both)) == pytest.approx((0.81, 0.0, 0.81))


def test_the_dimensions_are_one_input_and_a_decision_for_each():
    with pytest.raises(ValueError, match="one input and a `Noul`, `Score` or `Choice` output for each dimension; \\['kind'\\] are not"):
        Near(dspy.Signature("request -> heating").with_updated_fields("heating", type_=Noul).append("kind", dspy.OutputField(), type_=str))
    with pytest.raises(ValueError, match="one input and a `Noul`, `Score` or `Choice` output"):
        Near(About.append("more", dspy.InputField(), type_=str))
    with pytest.raises(ValueError, match="top must be at least 1"):
        Near(About, top=0)
    # The dimensions' input is the request's input of the same name.
    rdm = RDM("question, observation -> next: dspy.Tool", tools=steps(), fit=Near(About, top=2))
    rdm.set_lm(Client())
    with pytest.raises(ValueError, match="The dimensions take `request`, and the request has only \\['observation', 'question'\\]"):
        rdm(question="Cooler.")


def test_a_dimension_can_be_a_score_or_a_choice_and_its_parameters_are_read_when_a_label_is():
    class Sorted(dspy.Signature):
        """Say what a text is about."""

        request: str = dspy.InputField()
        heating: Noul = dspy.OutputField(desc="Is `inputs.request` about how warm it is?")
        urgency: Urgency = dspy.OutputField(desc="How soon does `inputs.request` need doing?")
        room: Room = dspy.OutputField(desc="Which room is `inputs.request` about?")

    near, client = Near(Sorted), Client()
    near.set_lm(client)
    # A score is how far up its rubric the text is, and a choice is a dimension for each option but "none of these".
    today, week = near.embed(["Cooler in the kitchen, today.", "The hall, some time this week."])
    assert today == {"heating": 0.9, "urgency": 1.0, "room.hall": 0.0, "room.kitchen": 1.0}
    assert week == {"heating": 0.0, "urgency": 0.5, "room.hall": 1.0, "room.kitchen": 0.0}
    # What is kept is what the model gave. Cuts give a score its level and weights multiply a choice's options:
    # both are the predictor's, read when a label is, so nothing is asked again.
    assert near.answers["The hall, some time this week."]["urgency"] == {"probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": 1.0}
    near.dimensions.fields["urgency"] = {"cuts": [1.5, 1.8]}
    near.dimensions.fields["room"] = {"weights": {"kitchen": 0.5}}
    today, week = near.embed(["Cooler in the kitchen, today.", "The hall, some time this week."])
    assert (today["urgency"], week["urgency"], today["room.kitchen"], week["room.hall"]) == (1.0, 0.0, 0.5, 1.0) and len(client.calls) == 2


def test_things_are_compared_by_the_measure_chosen():
    request, one, both = {"heating": 1.0, "lights": 0.0}, {"heating": 0.8, "lights": 0.0}, {"heating": 1.0, "lights": 1.0}
    scores = {measure: [round(Near(About, measure=measure).score(request, thing), 3) for thing in (one, both)] for measure in ("dot", "cosine", "euclidean", "manhattan", "jaccard")}
    assert scores == {
        "dot": [0.8, 1.0],  # what is shared, as much as both have of it: a dimension only the thing has costs it nothing
        "cosine": [1.0, 0.707],  # the direction shared: a dimension only the thing has counts against it
        "euclidean": [0.833, 0.5],  # one over one plus the distance
        "manhattan": [0.833, 0.5],
        "jaccard": [0.8, 0.5],  # what is shared over what either has
    }
    # A measure of your own is a function of the request's coordinates and a thing's.
    assert Near(About, measure=lambda request, thing: thing["lights"]).score(request, both) == 1.0
    with pytest.raises(ValueError, match=r"measure must be one of .*'dot'.*, or a function"):
        Near(About, measure="nearest")
    # The measure orders what is near. What shares nothing with the request is not near under any
    # measure, though a distance gives it a number: the lights are not offered for a request about heating.
    for measure in ("dot", "cosine", "euclidean", "manhattan", "jaccard"):
        client = Client()
        house(Near(About, top=3, measure=measure), client)(request="Two degrees cooler, please.")
        assert offered(client)[0] == ["cooler", "warmer", "finish"], measure
    # Where the order differs, the map is cut differently. A step about heating and a light is as near as any by the
    # dot product, so it is kept as the first on the map; by the cosine its other dimension puts it further off.
    tools = [dspy.Tool(lambda: "did both", name="both", desc="Make it warmer and turn on a light."), *steps()]
    for measure, first in (("dot", "both"), ("cosine", "cooler")):
        client = Client()
        rdm = RDM(Operate, tools=tools, fit=Near(About, top=1, measure=measure))
        rdm.set_lm(client)
        rdm(request="Two degrees cooler, please.")
        assert offered(client)[0] == [first, "finish"], measure


# A map cut to what is nearest a request


def test_a_pointer_is_offered_the_tools_nearest_the_request():
    client = Client()
    rdm = house(Near(About, top=3), client)
    result = rdm(request="Two degrees cooler, please.")
    # The two heating steps share a dimension with the request and the lights share none. `finish` is always offered.
    assert offered(client)[0] == ["cooler", "warmer", "finish"]
    assert result.observations["next"] == [["cooler", "did cooler"], ["finish", None]]
    # Each name on the map was asked once, and the request once: the second turn asks nothing again.
    assert len(client.asked("heating")) == 8 + 1 and len(client.asked("next")) == 2
    # What it asked are sub-calls of the run, for the answers only: in the trajectory, one level down.
    asked = [entry for entry in result.trajectory if "heating" in entry["outputs"]]
    assert len(asked) == 9 and {entry["depth"] for entry in asked} == {1} and all(entry["observations"] == {} for entry in asked)
    assert [entry["outputs"]["next"]["value"] for entry in result.trajectory if "next" in entry["outputs"]] == ["cooler", "finish"]
    # The next request asks about itself only, however long the map is.
    later = rdm(request="It is dim: a lamp, please.")
    assert len(client.asked("heating")) == 8 + 2 and len(later.trajectory) == 3
    assert offered(client)[2] == ["hall light", "kitchen light", "porch light", "finish"]


def test_what_near_asks_counts_against_the_runs_limits():
    client = Client()
    rdm = house(Near(About, top=3), client, max_predictor_calls=4)
    result = rdm(request="Two degrees cooler, please.")
    # The run's first request and three of the nine about dimensions were asked; a limit stopped the rest, and the run says so.
    assert len(client.calls) == 4 and result.termination_reason == "max_predictor_calls"
    # Labelling a long map is done before the run, outside it: then a run asks about its request only.
    near = Near(About, top=3)
    rdm = house(near, client, max_predictor_calls=4)
    near.embed([f"{tool.name}: {tool.desc}" for tool in rdm.tools.values()])
    assert rdm(request="Two degrees cooler, please.").termination_reason == "final"


def test_a_map_is_left_whole_where_it_is_short_or_nothing_sets_its_names_apart():
    client = Client()
    # Eight names and room for eight: nothing is asked.
    rdm = house(Near(About, top=8), client)
    rdm(request="Two degrees cooler, please.")
    assert client.asked("heating") == [] and len(offered(client)[0]) == 9
    # A request that shares nothing with any name has no direction, so every name is offered.
    rdm = house(Near(About, top=3), client)
    rdm(request="Do whatever is best.")
    assert len(offered(client)[-2]) == 9
    # The names the module always offers are not measured: a tool that takes no side, such as one that writes.
    rdm = house(Near(About, top=2), client, always=["hall light"])
    rdm(request="Warmer, by two degrees.")
    assert offered(client)[-2] == ["cooler", "warmer", "hall light", "finish"]


def test_a_request_can_say_how_it_uses_each_dimension():
    request = "A lamp would be nice, and cooler: it must be heating."
    # As it is, the request is as much about lights as about heating, and the three nearest include a light.
    client = Client()
    house(Near(About, top=3), client)(request=request)
    assert offered(client)[0] == ["hall light", "cooler", "warmer", "finish"]
    # Asked how it uses each dimension, it says heating is a filter and lights a boost: only the heating steps will do.
    client = Client()
    rdm = house(Near(About, top=3, use=True), client)
    result = rdm(request=request)
    assert offered(client)[0] == ["cooler", "warmer", "finish"]
    # That is one request more for each request, about its own text, kept as the labels are and counted in the run.
    assert len(client.asked("heating_use")) == 1 and client.asked("heating_use")[0]["inputs"] == {"request": request}
    assert len([entry for entry in result.trajectory if "heating_use" in entry["outputs"]]) == 1
    assert [name for name, _ in rdm.named_predictors()] == ["predict", "fit.dimensions", "fit.uses"]
    # A dimension the request ignores counts for nothing, though the request mentions nothing else.
    rdm(request="Do whatever is best.")
    assert len(offered(client)[-2]) == 9
    # A filter that would leave no name is let go: a house with no heating is offered its lights.
    client = Client()
    lights = RDM(Operate, tools=steps()[:6], fit=Near(About, top=3, use=True))
    lights.set_lm(client)
    lights(request=request)
    assert offered(client)[0] == ["hall light", "kitchen light", "porch light", "finish"]


def test_near_is_about_the_map_and_leaves_any_other_choice_alone():
    options = [(None, "None of these."), *((f"{room} lamp", "") for room in ("hall", "attic")), ("two degrees cooler", ""), ("one degree warmer", "")]
    pick = dspy.Signature("request -> step").with_updated_fields("step", type_=Choice[tuple(options)], desc="Which step does `inputs.request` ask for?")
    client = Client()
    rdm = RDM(pick, fit=Near(About, top=2))
    rdm.set_lm(client)
    rdm(request="Cooler, please.")
    # A Choice that is not the map is for `Fit`, where it is too long. Nothing was asked about dimensions.
    assert len(client.asked("step")[0]["questions"]["step"]) == 5 and client.asked("heating") == []


# Dimensions a language model writes


def test_a_dimension_is_a_name_and_a_question_that_can_be_written_while_the_module_runs():
    # With no dimension yet there is nothing to ask, and a map is offered whole.
    near, client = Near(Blank, top=2), Client()
    rdm = house(near, client)
    rdm(request="Two degrees cooler, please.")
    assert near.questions == {} and near.embed(["Cooler."]) == [{}] and client.asked("heating") == []
    assert len(offered(client)[0]) == 9
    # A dimension is written as a name and a question. The names on the map are asked it when next they are needed.
    assert near.add("heating", "Is `inputs.request` about how warm it is?") == "`heating` is now a dimension: Is `inputs.request` about how warm it is?"
    rdm(request="Two degrees cooler, please.")
    assert offered(client)[-2] == ["cooler", "warmer", "finish"] and len(client.asked("heating")) == 9
    # Only a text that lacks an answer is asked. With another dimension, each text is asked once more.
    near.add("lights", "Is `inputs.request` about a light?")
    assert near.embed(["Two degrees cooler, please.", "A lamp, and cooler."]) == [{"heating": 0.9, "lights": 0.0}, {"heating": 0.9, "lights": 0.8}]
    assert len(client.asked("lights")) == 2 and len(client.asked("heating")) == 9 + 2
    # The questions are plain data; a name there already is takes the new question, and what was answered to the old one goes.
    assert near.questions == {"heating": "Is `inputs.request` about how warm it is?", "lights": "Is `inputs.request` about a light?"}
    assert near.add("lights", "Is `inputs.request` about a lamp?") == "`lights` now asks: Is `inputs.request` about a lamp?"
    assert near.answers["A lamp, and cooler."] == {"heating": 0.9}
    # Removing a dimension removes its answers and its threshold with it.
    near.dimensions.fields["lights"] = {"threshold": 0.7}
    assert near.remove("lights") == "`lights` is no longer a dimension." and near.dimensions.fields == {}
    assert near.embed(["A lamp, and cooler."]) == [{"heating": 0.9}] and near.remove("lights").startswith("There is no dimension `lights`")
    for name, question in (("two words", "A question?"), ("request", "A question?"), ("empty", " ")):
        with pytest.raises(ValueError, match="A dimension is a name, written as one word and not `request`, and a question"):
            near.add(name, question)


def test_a_language_model_writes_a_dimension_through_its_tools_and_the_decision_model_applies_it():
    in_process = pytest.importorskip("dspy_interpreters").InProcessInterpreter
    near = Near(About, top=2)
    # `add` is a plain function, so it is a tool of a dspy.RLM, which writes the name and the question in code.
    labeler = dspy.RLM("request -> made", tools=[near.add], interpreter_factory=in_process, max_iters=2)
    code = "add(name='doors', question='Is `inputs.request` about a door or its lock?')\nSUBMIT(made='doors')"
    labeler.set_lm(DummyLM([{"reasoning": "No dimension is about doors.", "code": f"```python\n{code}\n```"}]))
    doors = [dspy.Tool(lambda name=name: f"did {name}", name=name, desc=desc) for name, desc in (("close door", "Close the front door."), ("lock door", "Lock the front door."))]
    label = dspy.Tool(labeler, name="label", desc="Have a new label made: nothing listed is near what is asked.")
    client = Client("label", "lock door", "finish")
    rdm = RDM(Operate, tools=[*steps(), *doors, label], fit=near, always=["label"])
    rdm.set_lm(client)
    result = rdm(request="Lock the front door.")
    # No dimension was about a door, so the request shared nothing with any name and the whole map was offered.
    assert len(offered(client)[0]) == 12
    # The decision model pointed at `label`, and the language model wrote a dimension: its name and its question.
    assert near.questions["doors"] == "Is `inputs.request` about a door or its lock?"
    # The decision model then said of every name on the map whether the new label applies, and the map was cut by it.
    assert offered(client)[1] == ["close door", "lock door", "label", "finish"]
    assert result.observations["next"][1] == ["lock door", "did lock door"] and result.termination_reason == "final"
    # The change is in the trajectory, on the request whose tool made it, and `show` is what a writer reads first.
    assert result.trajectory[0]["changes"] == [{"by": "near", "did": "add", "name": "doors"}]
    assert near.show().splitlines()[-1] == "doors: Is `inputs.request` about a door or its lock?"


# Thresholds and ReAnchor


def test_a_label_is_the_probability_until_its_dimension_has_a_threshold_and_a_threshold_asks_nothing_again():
    near, client = Near(About), Client()
    near.set_lm(client)
    assert near.embed(["It is dim and a bit stuffy."]) == [{"heating": 0.6, "lights": 0.5}]
    # With a threshold, an answer under it counts as nothing. It is read when the label is, so nothing is asked again.
    near.dimensions.fields["heating"] = {"threshold": 0.75}
    assert near.embed(["It is dim and a bit stuffy."]) == [{"heating": 0.0, "lights": 0.5}] and len(client.calls) == 1
    near.dimensions.fields["heating"] = {"threshold": 0.6}
    assert near.embed(["It is dim and a bit stuffy."]) == [{"heating": 0.6, "lights": 0.5}] and len(client.calls) == 1
    assert near.answers == {"It is dim and a bit stuffy.": {"heating": 0.6, "lights": 0.5}}


def test_labels_are_one_models_answers_and_are_dropped_when_another_is_asked():
    near, first, second = Near(About), Client(), Client()
    second.model = "jev-next"
    near.set_lm(first)
    near.embed(["Make it cooler."])
    near.embed(["Make it cooler."])
    assert len(first.calls) == 1 and near.labels["model"] == "jev-test"
    # Another model would answer otherwise, so what the first answered is not read as its own.
    near.set_lm(second)
    near.embed(["Make it cooler."])
    assert len(second.calls) == 1 and near.labels["model"] == "jev-next" and list(near.answers) == ["Make it cooler."]


@pytest.mark.parametrize("by_hand", [False, True])
def test_reanchor_fits_the_threshold_of_a_dimension_against_the_whole_program(by_hand):
    # "Dim and stuffy" is 0.6 about heating and 0.5 about lights, so the one name offered is a heating step,
    # where a light was wanted. Nothing about the answer itself can be fitted: the cut that helps is a dimension's.
    rdm = house(Near(About, top=1))
    if by_hand:
        rdm.fit.dimensions.fields["heating"] = {"threshold": 0.5}  # a threshold there already is, and answers already kept
    assert rdm(request="It is dim and a bit stuffy.").observations["next"][0][0] == "cooler"
    trainset = [
        dspy.Example(request=request, first=first).with_inputs("request")
        for request, first in [("It is dim and a bit stuffy.", "hall light"), ("Dim, and stuffy too.", "hall light"), ("A lamp, please.", "hall light"), ("Cooler, please.", "cooler")]
    ]
    optimizer = ReAnchor(lambda example, prediction, trace=None: float(prediction.observations["next"][0][0] == example.first))
    tuned = optimizer.compile(rdm, trainset=trainset)
    # The cut is a parameter of the predictor that asks the dimensions: named, saved, and fitted against the metric.
    assert (optimizer.report["train_score_before"], optimizer.report["train_score"]) == (0.5, 1.0)
    assert 0.6 < tuned.fit.dimensions.fields["heating"]["threshold"] < 0.9
    assert rdm.fit.dimensions.fields == ({"heating": {"threshold": 0.5}} if by_hand else {})
    assert ("fit.dimensions", "heating") in [(row["predictor"], row["field"]) for row in optimizer.report["fitted"] if "skipped" not in row]
    # The tuned copy has its own labels, and reads them under its own thresholds.
    assert tuned(request="It is dim and a bit stuffy.").observations["next"][0][0] == "hall light"
    assert tuned.fit.answers is not rdm.fit.answers and tuned.fit.embed(["It is dim and a bit stuffy."]) == [{"heating": 0.0, "lights": 0.5}]


# A program that is saved, loaded and copied


def test_the_labels_are_saved_and_loaded_with_the_program(tmp_path):
    def program(signature):
        return RDM(Operate, tools=steps(), fit=Near(signature, top=2))

    rdm, client = program(Blank), Client()
    rdm.fit.add("heating", "Is `inputs.request` about how warm it is?")
    rdm.fit.add("doors", "Is `inputs.request` about a door or its lock?")
    rdm.fit.dimensions.fields["doors"] = {"threshold": 0.7}
    with dspy.context(lm=client):
        rdm(request="Two degrees cooler, please.")
    rdm.save(tmp_path / "program.json")
    # A program built the same way is given the dimensions that were written, their thresholds, and every answer.
    fresh, again = program(Blank), Client()
    fresh.load(tmp_path / "program.json")
    assert fresh.fit.questions == rdm.fit.questions and list(fresh.fit.dimensions.signature.output_fields) == ["heating", "doors"]
    assert fresh.fit.dimensions.fields == {"doors": {"threshold": 0.7}} and fresh.fit.answers == rdm.fit.answers
    # So it asks nothing about the map again: the first request is the one that points.
    with dspy.context(lm=again):
        fresh(request="Two degrees cooler, please.")
    assert offered(again)[0] == ["cooler", "warmer", "finish"] and again.asked("heating") == []
    # A program built with other dimensions has, once loaded, the ones that were saved.
    other = program(About)
    other.load(tmp_path / "program.json")
    assert list(other.fit.questions) == ["heating", "doors"]
    # A copy of the program has labels of its own.
    twin = rdm.deepcopy()
    twin.fit.add("lights", "Is `inputs.request` about a light?")
    assert list(twin.fit.questions) == ["heating", "doors", "lights"] and list(rdm.fit.questions) == ["heating", "doors"]
    assert twin.fit.answers == {} or twin.fit.answers is not rdm.fit.answers


def test_the_coordinates_are_an_embedder_for_dspys_own_search():
    pytest.importorskip("numpy")
    near, client = Near(About), Client()
    near.set_lm(client)
    documents = ["The lamp in the hall is out.", "Make it two degrees cooler.", "Nothing of either."]
    # Each text is a row of numbers, one for each coordinate, so DSPy's retriever searches by named dimensions.
    embedder = near.embedder()
    assert embedder(documents[:2]).tolist() == [pytest.approx([0.0, 0.8]), pytest.approx([0.9, 0.0])]
    search = dspy.retrievers.Embeddings(corpus=documents, embedder=embedder, k=1)
    assert search("It is stuffy in here.").passages == ["Make it two degrees cooler."]
    assert search("It is dim.").passages == ["The lamp in the hall is out."]


# With other helpers


def test_several_fits_are_applied_in_order_and_each_keeps_its_predictors():
    client = Client()
    rdm = RDM(Operate, tools=steps(), fit=[Near(About, top=2), Fit(max_options=4)], always=["hall light"])
    rdm.set_lm(client)
    rdm(request="Two degrees cooler, please.")
    # Near cut the map to what is nearest, and Fit was given what Near left, which fits. Both leave what is always offered.
    assert offered(client)[0] == ["cooler", "warmer", "hall light", "finish"]
    # Both are reached as the program's predictors, so both are given the client, saved, and fitted.
    assert [name for name, _ in rdm.named_predictors()] == ["predict", "fit[0].dimensions", "fit[1].needed"]
    # The other way round, Fit's own questions about a long map are left alone by Near, and the run still ends.
    client = Client()
    rdm = RDM(Operate, tools=steps(), fit=[Fit(max_options=4), Near(About, top=2)], always=["hall light"])
    rdm.set_lm(client)
    result = rdm(request="Two degrees cooler, please.")
    assert result.termination_reason == "final" and all("finish" in options and "hall light" in options for options in offered(client))
