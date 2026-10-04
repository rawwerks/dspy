"""Tests for RDM.Filter: sorting what a decision model rated by its probabilities, and keeping the likeliest."""

import pytest

import dspy
from dspy.experimental import RDM, Choice, Noul, ReAnchor

Filter = RDM.Filter
Way = Choice[("north", ""), ("south", ""), ("east", ""), (None, "No way.")]


def test_it_sorts_by_probability_and_keeps_the_likeliest():
    # With nothing set it only sorts. What it gives is every thing with its probability, likeliest first.
    assert Filter()({"east": 0.1, "north": 0.5, "south": 0.4}) == [("north", 0.5), ("south", 0.4), ("east", 0.1)]
    # top: so many at most, a setting like the width of a beam.
    assert Filter(top=2)({"north": 0.5, "south": 0.4, "east": 0.1}) == [("north", 0.5), ("south", 0.4)]
    # Things rated alike keep the order they were given in.
    assert [thing for thing, _ in Filter()({"a": 0.3, "b": 0.3, "c": 0.9})] == ["c", "a", "b"]
    assert Filter(top=2)({}) == []
    with pytest.raises(ValueError, match="top must be at least 1"):
        Filter(top=0)


def test_it_reads_what_was_rated_from_the_answers_a_model_gave():
    # A Choice: every option with its probability, each as the value it was declared with.
    answer = Way(value="north", confidence=0.5, probabilities={"north": 0.5, "south": 0.3, "east": 0.05, "None": 0.15})
    assert Filter(top=2)(answer) == [("north", 0.5), ("south", 0.3)]
    assert Filter()(answer)[-2:] == [(None, 0.15), ("east", 0.05)]
    # A prediction: its yes or no answers that are a yes, by the name of the output.
    prediction = dspy.Prediction(holds_0=Noul(value=False, confidence=0.6, probability=0.2), holds_1=Noul(value=True, confidence=0.8, probability=0.9), way=answer)
    assert Filter()(prediction) == [("holds_1", 0.9)]
    # Pairs: anything, each with the yes or no given about it. Only the yeses are kept.
    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    verdicts = [Noul(value=False, confidence=0.8, probability=0.1), Noul(value=True, confidence=0.4, probability=0.7), Noul(value=True, confidence=0.2, probability=0.6)]
    assert Filter()(zip(records, verdicts, strict=True)) == [({"id": 2}, 0.7), ({"id": 3}, 0.6)]
    # A yes that came without a probability counts as certain.
    assert Filter()([("a", Noul(value=True, confidence=1.0)), ("b", Noul(value=False, confidence=1.0))]) == [("a", 1.0)]
    with pytest.raises(TypeError, match="cannot be read as a probability"):
        Filter()([("a", "likely")])


class Client:
    """A stand-in System One client: `answer(inputs, name, question)` gives each question's probabilities."""

    supports_decision_requests = True
    cache = True
    model = "jev-test"

    def __init__(self, answer):
        self.answer, self.calls, self.history, self.callbacks = answer, [], [], []

    def __call__(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        answers = {}
        for name, question in questions.items():
            answer = self.answer(state["inputs"], name, question)
            answers[name] = {"noul": answer} if question["type"] == "noul" else {"probabilities": answer, "confidence": max(answer.values())}
        return answers


TREE = {"root": ["pets", "furniture", "garden"], "pets": ["cat bed", "dog bed"], "furniture": ["shelf", "sofa"], "garden": ["hose", "spade"]}


def under(node):
    options = Choice[tuple((child, "") for child in TREE[node])]
    return dspy.Signature("listing -> category").with_updated_fields("category", type_=options, desc="Which category fits `inputs.listing`?")


class Fits(dspy.Signature):
    """Judge one leaf."""

    listing: str = dspy.InputField()
    leaf: str = dspy.InputField()
    fits: Noul = dspy.OutputField(desc="Does `inputs.leaf` fit `inputs.listing`?")


def category(listing, call: RDM.Call):
    """Follow the two likeliest categories, each to its leaves; then keep the leaves that fit, likeliest first."""
    leaves = []
    for option, _ in Filter(top=2)(call.prediction.category):
        leaves += call.rdm(signature=under(option), listing=listing).observations["category"] if option in TREE else [option]
    if call.depth:
        return leaves
    verdicts = call.rdm.batch([{"signature": Fits, "listing": listing, "leaf": leaf, "max_depth": 0} for leaf in leaves])
    return Filter()(zip(leaves, (verdict.fits for verdict in verdicts), strict=True))


def odds(fits):
    def answer(inputs, name, question):
        if name == "fits":
            return fits.get(inputs["leaf"], 0.1)
        rated = {"pets": 0.5, "furniture": 0.4, "garden": 0.1, "cat bed": 0.8, "dog bed": 0.2, "shelf": 0.7, "sofa": 0.3, "hose": 0.6, "spade": 0.4}
        return {option: rated[option] for option in question["criteria"]}

    return answer


def test_a_tool_follows_the_likeliest_options_and_keeps_what_a_batch_said_yes_to():
    rdm = RDM(under("root"), tools=[category])
    client = Client(odds({"cat bed": 0.9, "shelf": 0.6}))
    rdm.set_lm(client)
    result = rdm(listing="A padded window shelf for a cat.")
    # Two of the three branches were followed, each to its two likeliest leaves, and the third, the garden,
    # was not. Of those four leaves, the two said yes of were kept, likeliest first.
    assert [list(call["questions"]["category"]["criteria"]) for call in client.calls if "category" in call["questions"]] == [
        ["pets", "furniture", "garden"], ["cat bed", "dog bed"], ["shelf", "sofa"],
    ]
    assert result.observations["category"] == [("cat bed", 0.9), ("shelf", 0.6)]


def test_the_cut_is_the_predictors_threshold_so_reanchor_fits_it():
    # The model says 0.45 of the leaf that fits and 0.3 of one that does not: at one half, nothing is a yes.
    rdm = RDM(under("root"), tools=[category])
    rdm.fits = dspy.Predict(Fits)
    rdm.set_lm(Client(odds({"cat bed": 0.45, "shelf": 0.3})))
    assert rdm(listing="A padded window shelf for a cat.").observations["category"] == []
    # The filter has no cut of its own to tune. The yes is the predictor's, and the predictor is the program's.
    trainset = [dspy.Example(listing=f"A padded window shelf for cat {n}.").with_inputs("listing") for n in range(6)]
    optimizer = ReAnchor(lambda example, prediction, trace=None: float([leaf for leaf, _ in prediction.observations.get("category", [])] == ["cat bed"]))
    tuned = optimizer.compile(rdm, trainset=trainset)
    assert (optimizer.report["train_score_before"], optimizer.report["train_score"]) == (0.0, 1.0)
    assert 0.3 < tuned.fits.fields["fits"]["threshold"] < 0.45
