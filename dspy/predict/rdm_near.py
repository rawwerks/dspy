"""
Near: gives the names on a map labels, and offers the names nearest a request.

The names on a `dspy.experimental.RDM`'s map have one dimension, a description, and the decision
model reads every one each time it points. `Near` gives each name labels as well: its answers
to the questions of a signature, each question a dimension with a name.

    class About(dspy.Signature):
        \"\"\"Say what a text is about.\"\"\"

        task: str = dspy.InputField()
        heating: Noul = dspy.OutputField(desc="Is `inputs.task` about how warm it is?")
        lights: Noul = dspy.OutputField(desc="Is `inputs.task` about a light?")

    rdm = RDM(Operate, tools=steps, fit=RDM.Near(About, top=8))

    "cool by degrees: Make it cooler by a number of degrees."    {"heating": 0.97, "lights": 0.0}
    "kitchen lights off: Turn the kitchen lights off."            {"heating": 0.0, "lights": 0.99}
    "It is too warm in here."                                     {"heating": 0.91, "lights": 0.0}

A name is asked once, when it is first on the map, and what it answered is kept: `near.answers`
holds what the model gave for each text and dimension. It is part of the program, so `save`
writes it and a copy has its own. Each request is asked the same questions, and the pointer is
offered the `top` names nearest it. Every number has a name, so it can be read why a name was
offered.

A dimension is a yes or no question (`Noul`), a `Score`, or a `Choice`, whose options are a
dimension each. A dimension is also a name and a question that can be written while the module
runs: `near.add(name, question)` and `near.remove(name)` are plain functions, so a language
model writes dimensions as it does anything else, as the tools of a `dspy.RLM`, say. The
language model coins the label; the decision model says of each thing whether it applies.

`Near` has no threshold or weight of its own. A label is what the model answered, until its
dimension is given a parameter on `near.dimensions`, the predictor that asks: a threshold,
under which a yes or no counts as nothing; cuts, which give a score its level; weights, which
multiply a choice's options. Those are parameters of the program, saved with it and fitted by
ReAnchor, and they are read each time a label is, so changing one asks nothing again.
"""

import json
import math
from typing import Any, Callable

from dspy.adapters.types.decision import Choice, Noul, Score
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.predict.rdm_shared import POINTS, Kept, changed, likely, stays
from dspy.predict.rdm_shared import options as _options
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.signatures.field import OutputField
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.annotation import experimental

# How a request uses a label, where a `Near` asks: what the request is asked of each dimension.
_USE = Choice[("filter", "Only what has this label will do."), ("boost", "What has this label is better."), ("ignore", "This label does not matter here.")]


@experimental
class Near(Module):
    """Labels for the names on a map, and the names nearest a request.

    Given to a ``dspy.experimental.RDM`` as ``fit=RDM.Near(signature)``, it cuts an output that
    points at a tool, where the map has more than ``top`` names, to the ``top`` nearest the
    request, in the order they had. ``finish`` and the names the module always offers stay. A
    name that shares nothing with the request is not offered, and where no name shares
    anything, the request is asked as it is. What it asks are sub-calls of the run, so they
    count against its limits and are in its trajectory.

    Used alone, ``near.embed(texts)`` gives each text's coordinates and ``near.score`` compares
    two of them by the ``measure``; ``near.embedder()`` is the same as a ``dspy.Embedder``, for
    DSPy's own search over documents.

    Args:
        signature: The dimensions: one input, and an output for each, a ``Noul``, a ``Score``
            or a ``Choice``. The input is given a request's input of the same name, and each
            name on the map as ``"name: description"``, so name it after the request's input
            that says what is wanted. It may have no outputs yet, where every dimension is to
            be written with ``add``.
        top: How many names a request offers at most, beside those always offered.
        measure: How two sets of coordinates are compared, the larger the nearer: ``"dot"``
            (what they share, each dimension as much as both have of it), ``"cosine"`` (the
            direction they share, so that a dimension only one of them has counts against
            them), ``"euclidean"`` or ``"manhattan"`` (one over one plus the distance),
            ``"jaccard"`` (what they share over what either has), or a function of the
            request's coordinates and a thing's.
        use: Whether a request is also asked how it uses each dimension: as a filter, so that
            only names that have it are offered; as a boost; or not at all. One request more
            for each request. A filter that would leave no name is let go.

    Examples:
        ```python
        import dspy
        from dspy.experimental import RDM, Noul, Score, TypeSafe

        Urgency = Score["whenever", "this week", "today"]

        class About(dspy.Signature):
            \"\"\"Say what a text is about.\"\"\"

            task: str = dspy.InputField()
            billing: Noul = dspy.OutputField(desc="Is `inputs.task` about a payment or an invoice?")
            urgency: Urgency = dspy.OutputField(desc="How soon does `inputs.task` need doing?")

        near = RDM.Near(About, top=5)
        rdm = RDM(Triage, tools=tools, fit=near)   # tools: more than five
        rdm.set_lm(TypeSafe("jev-latest"))
        rdm(task="I was charged twice this month, please fix it today.")

        near.dimensions.fields["billing"] = {"threshold": 0.7}   # or let ReAnchor fit it
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        top: int = 8,
        measure: str | Callable[[dict[str, float], dict[str, float]], float] = "dot",
        use: bool = False,
    ):
        super().__init__()
        signature = ensure_signature(signature)
        other = [name for name, output in signature.output_fields.items() if _kind(output) is None]
        if len(signature.input_fields) != 1 or other:
            raise ValueError(
                "The dimensions are a signature with one input and a `Noul`, `Score` or `Choice` output for each dimension"
                + (f"; {other} are not." if other else ".")
            )
        if top < 1:
            raise ValueError("top must be at least 1: a request has to offer something.")
        if not callable(measure) and measure not in MEASURES:
            raise ValueError(f"measure must be one of {sorted(MEASURES)}, or a function of two sets of coordinates; got {measure!r}.")
        self.top = top
        self.measure = measure
        self.use = use
        self._input = next(iter(signature.input_fields))
        self._asks = signature  # the signature the dimensions were given as: its input, its instructions, and the type of each output
        # Every dimension's question; every text's answers, as the model gave them; how each request
        # uses each dimension; and the model that gave them all.
        self.labels = {
            "questions": {name: output.json_schema_extra.get("desc", "") for name, output in signature.output_fields.items()},
            "answers": {},
            "uses": {},
            "model": None,
        }
        # What makes the labels part of the program. It comes before the predictors, so that when a
        # program is loaded the dimensions are there before a predictor's own state is read.
        self.kept = Kept(self.labels, loaded=self._sync)
        # The questions that make the dimensions, as a predictor: what holds each one's threshold, cuts or weights.
        self.dimensions = Predict(signature)
        # How a request uses each dimension, as a predictor too: what holds the weights of filter, boost and ignore.
        self.uses = Predict(self._uses()) if use else None

    @property
    def questions(self) -> dict[str, str]:
        """Every dimension's question, by name."""
        return dict(self.labels["questions"])

    @property
    def answers(self) -> dict[str, dict]:
        """What each text asked so far answered, by dimension, as the model gave it.

        They are the answers of one model, and are dropped when another is asked.
        """
        return self.labels["answers"]

    def add(self, name: str, question: str) -> str:
        """Add a dimension, a yes or no question, or put another question under a name there already is.

        Args:
            name: The dimension's name: one word, as a Python name is written.
            question: A yes or no question about a text, which is referred to as the
                dimensions' input is (``inputs.task``, say).
        """
        name, question = str(name).strip(), str(question).strip()
        if not name.isidentifier() or name == self._input or not question:
            raise ValueError(f"A dimension is a name, written as one word and not `{self._input}`, and a question; got {name!r} and {question!r}.")
        known = name in self.labels["questions"]
        if self.labels["questions"].get(name) != question:
            self.labels["questions"][name] = question
            self._forget(name)  # what was answered was the answer to another question
            self._sync()
            changed({"by": "near", "did": "add", "name": name})
        return f"`{name}` now asks: {question}" if known else f"`{name}` is now a dimension: {question}"

    def remove(self, name: str) -> str:
        """Remove a dimension, with what was answered for it and any parameter it was given.

        Args:
            name: The dimension's name.
        """
        if self.labels["questions"].pop(name, None) is None:
            return f"There is no dimension `{name}`: the dimensions are {list(self.labels['questions'])}."
        self._forget(name)
        self._sync()
        changed({"by": "near", "did": "remove", "name": name})
        return f"`{name}` is no longer a dimension."

    def show(self) -> str:
        """Every dimension there is: its name and its question."""
        return "\n".join(f"{name}: {question}" for name, question in self.labels["questions"].items()) or "There is no dimension yet."

    def forward(self, signature, inputs, call=None):
        """The signature with its output that points at a tool cut to the names nearest the request."""
        for name, output in signature.output_fields.items():
            if not output.json_schema_extra.get(POINTS) or not self.labels["questions"]:
                continue
            options = _options(output)
            always = [option for option in options if stays(output, option)]
            measured = [option for option in options if option not in always]
            if len(measured) <= self.top:
                continue
            if self._input not in inputs:
                raise ValueError(
                    f"The dimensions take `{self._input}`, and the request has only {sorted(inputs)}: name the dimensions' "
                    "input after the request's input that says what is wanted."
                )
            asked = _text(inputs[self._input])
            request, *things = self.embed([asked, *(_text(*option) for option in measured)], call)
            filters = []
            if self.use:
                weights, filters = self._used(asked, call)
                request = {key: value * weights.get(key.split(".")[0], 1.0) for key, value in request.items()}
            # What shares nothing with the request is not near it at all, however it is measured.
            near = [position for position, thing in enumerate(things) if _dot(request, thing) > 0]
            # A dimension the request filters on: only a name that shares it will do. A filter that leaves nothing is let go.
            for dimension in filters:
                part = {key: value for key, value in request.items() if key.split(".")[0] == dimension}
                near = [position for position in near if _dot(part, things[position]) > 0] or near
            if not near:
                continue  # nothing sets the names apart
            # The nearest, in the order they had.
            scores = {position: self.score(request, things[position]) for position in near}
            nearest = sorted(sorted(near, key=lambda position: -scores[position])[: self.top])
            signature = signature.with_updated_fields(name, type_=Choice[(*(measured[position] for position in nearest), *always)])
        return signature, inputs

    def embed(self, texts: list[str], call: Any = None) -> list[dict[str, float]]:
        """The coordinates of each text: a number for every yes or no dimension and every score, and one for each option of a choice.

        A text is asked once, and only a text that lacks an answer is asked. Inside a run, when
        given the ``call``, what is asked are sub-calls of it. A dimension's threshold, cuts or
        weights are read now, so changing one asks nothing again.
        """
        questions, answers = self.labels["questions"], self.labels["answers"]
        if not questions:
            return [{} for _ in texts]  # no dimension yet: nothing to ask
        self._same_model()
        new = list(dict.fromkeys(text for text in texts if not set(questions) <= set(answers.get(text, {}))))
        for text, answered in zip(new, self._ask(self.dimensions, new, call), strict=True):
            if all(answered[name] is not None for name in questions):  # a request a limit stopped answers nothing
                answers[text] = {name: _given(answered[name]) for name in questions}
        self._decoded(self.dimensions, [answers[text] for text in texts if text not in new and text in answers])
        return [self._coordinates(answers.get(text, {})) for text in texts]

    def embedder(self, **kwargs):
        """These coordinates as a ``dspy.Embedder``: each text a row of numbers, one for each coordinate, in a fixed order.

        For DSPy's own search, ``dspy.retrievers.Embeddings(corpus=texts, embedder=near.embedder())``.
        It embeds the corpus once, so build it after the dimensions and their parameters are settled.
        """
        from dspy.clients.embedding import Embedder

        def rows(texts, **_):
            return [list(coordinates.values()) for coordinates in self.embed(list(texts))]

        return Embedder(rows, caching=False, **kwargs)

    def score(self, request: dict[str, float], thing: dict[str, float]) -> float:
        """How near a thing is to a request, by the measure chosen: the larger, the nearer."""
        return (self.measure if callable(self.measure) else MEASURES[self.measure])(request, thing)

    def _ask(self, predictor, texts, call):
        """Each text's answers to a predictor's questions."""
        if not texts:
            return []
        if call is not None:
            # In a run: sub-calls, for the answers only, which the module puts to this predictor.
            return call.rdm.batch([{"signature": predictor.signature, self._input: text, "max_depth": 0} for text in texts])
        return predictor.batch([Example(**{self._input: text}).with_inputs(self._input) for text in texts], max_errors=0, disable_progress_bar=True)

    def _coordinates(self, answered):
        """A text's coordinates from what it answered, under the parameters of the dimensions as they are now."""
        coordinates = {}
        for name, output in self.dimensions.signature.output_fields.items():
            kind, given, config = _kind(output), answered.get(name), self.dimensions.fields.get(name, {})
            if kind is Choice:
                # A dimension for each option, as likely as the option was, times its weight where it has one.
                weights = config.get("weights") or {}
                for label in output.annotation.criteria():
                    if label != "None":
                        coordinates[f"{name}.{label}"] = (given or {}).get("probabilities", {}).get(label, 0.0) * weights.get(label, 1.0)
            elif kind is Score:
                # How far up the rubric: the level its cuts give it where it has cuts, the mean level otherwise.
                levels, value, cuts = len(output.annotation.criteria()), _mean(given), config.get("cuts")
                coordinates[name] = sum(value >= cut for cut in cuts) / len(cuts) if cuts else value / max(levels - 1, 1)
            else:
                # How likely a yes is, and nothing under its threshold where it has one.
                probability, threshold = given or 0.0, config.get("threshold")
                coordinates[name] = probability if threshold is None or probability >= threshold else 0.0
        return coordinates

    def _used(self, text, call):
        """How a request uses each dimension: how much it counts, and which dimensions are filters."""
        uses, asked = self.labels["uses"], [f"{name}_use" for name in self.labels["questions"]]
        if not set(asked) <= set(uses.get(text, {})):
            answered = self._ask(self.uses, [text], call)[0]
            if any(answered[name] is None for name in asked):
                return {}, []  # a limit stopped the request: every dimension counts as it is
            uses[text] = {name: _given(answered[name]) for name in asked}
        else:
            self._decoded(self.uses, [uses[text]])
        weights, filters = {}, []
        for name in self.labels["questions"]:
            likely, weight = uses[text][f"{name}_use"]["probabilities"], self.uses.fields.get(f"{name}_use", {}).get("weights") or {}
            scores = {label: probability * weight.get(label, 1.0) for label, probability in likely.items()}
            # A dimension counts as far as the request does not ignore it, and is a filter where that is its likeliest use.
            weights[name] = 1.0 - scores.get("ignore", 0.0) / (sum(scores.values()) or 1.0)
            if max(scores, key=scores.get) == "filter":
                filters.append(name)
        return weights, filters

    def _decoded(self, predictor, kept):
        """Record answers read from what was kept as evidence a predictor decoded, for whatever is collecting it.

        ReAnchor fits a parameter from what it sees decoded while the program runs. An answer
        that was kept is decoded here and not by the predictor, so it is recorded here.
        """
        recorder = settings.get("_decision_evidence")
        if recorder is not None and kept:
            evidence = []
            for answered in kept:
                for name, given in answered.items():
                    if isinstance(given, dict) and "probabilities" in given:
                        levels = _kind(predictor.signature.output_fields[name]) is Score
                        given = {**given, "probabilities": {int(key) if levels else key: value for key, value in given["probabilities"].items()}}
                    elif not isinstance(given, dict):
                        given = {"noul": given}
                    evidence.append((predictor, name, given))
            log, lock = recorder
            with lock:
                log.extend(evidence)

    def _same_model(self):
        """Drop what another model answered: a label is one model's answer."""
        model = getattr(self.dimensions.lm or settings.lm, "model", None)
        if model is not None:
            if self.labels["model"] not in (None, model):
                self.labels["answers"].clear()
                self.labels["uses"].clear()
            self.labels["model"] = model

    def _forget(self, name):
        """Drop what was answered for a dimension, and its parameters where the dimension is gone."""
        for answered in self.labels["answers"].values():
            answered.pop(name, None)
        self.labels["uses"].clear()
        if name not in self.labels["questions"]:
            self.dimensions.fields.pop(name, None)
            if self.uses is not None:
                self.uses.fields.pop(f"{name}_use", None)

    def _sync(self):
        """Make the predictors ask the questions the labels hold: after one is written, and after a program is loaded."""
        signature, questions = self._asks, self.labels["questions"]
        for name in list(signature.output_fields):
            if name not in questions:
                signature = signature.delete(name)
        for name, question in questions.items():
            if name not in signature.output_fields:
                signature = signature.append(name, OutputField(desc=question), type_=Noul)  # a dimension that was written
            elif signature.output_fields[name].json_schema_extra.get("desc") != question:
                signature = signature.with_updated_fields(name, desc=question)
        self.dimensions.signature = signature
        if self.uses is not None:
            self.uses.signature = self._uses()

    def _uses(self):
        """The questions a request is asked about how it uses each dimension."""
        signature = self._asks.with_instructions("Say how a request uses each label of the things it could be given.")
        for name in list(signature.output_fields):
            signature = signature.delete(name)
        for name, question in self.labels["questions"].items():
            asks = f"How does `inputs.{self._input}` use the label `{name}`, which asks of a thing: {question}"
            signature = signature.append(f"{name}_use", OutputField(desc=asks), type_=_USE)
        return signature


def _kind(output):
    """Which kind of dimension an output is: Noul, Score or Choice. None for any other output."""
    annotation = output.annotation
    return next((kind for kind in (Noul, Score, Choice) if isinstance(annotation, type) and issubclass(annotation, kind)), None)


def _given(answer):
    """An answer as it is kept: the probability of a yes, or the probabilities of a score's levels or a choice's options."""
    if not isinstance(answer, (Score, Choice)):
        return likely(answer)
    return {"probabilities": {str(key): value for key, value in answer.probabilities.items()}, "confidence": answer.confidence}


def _mean(given):
    """The mean level of a score as it was kept."""
    if not given:
        return 0.0
    levels = given["probabilities"]
    return sum(int(level) * probability for level, probability in levels.items()) / (sum(levels.values()) or 1.0)


def _dot(request, thing):
    """The dot product: what the two share, each dimension as much as both have of it. A dimension only one of them has costs nothing."""
    return sum(value * thing.get(name, 0.0) for name, value in request.items())


def _cosine(request, thing):
    """The cosine: the direction the two share, whatever their lengths. A dimension only one of them has counts against them."""
    lengths = math.sqrt(_dot(request, request)) * math.sqrt(_dot(thing, thing))
    return _dot(request, thing) / lengths if lengths else 0.0


def _euclidean(request, thing):
    """One over one plus the straight-line distance: a dimension only one of the two has counts against them."""
    return 1.0 / (1.0 + math.sqrt(sum((value - thing.get(name, 0.0)) ** 2 for name, value in request.items())))


def _manhattan(request, thing):
    """One over one plus the distance summed a dimension at a time: as euclidean, with large differences counted no more than small ones."""
    return 1.0 / (1.0 + sum(abs(value - thing.get(name, 0.0)) for name, value in request.items()))


def _jaccard(request, thing):
    """What the two share over what either has (Tanimoto): for coordinates that are close to yes or no, the share of dimensions in common."""
    either = sum(max(value, thing.get(name, 0.0)) for name, value in request.items())
    return sum(min(value, thing.get(name, 0.0)) for name, value in request.items()) / either if either else 0.0


MEASURES = {"cosine": _cosine, "dot": _dot, "euclidean": _euclidean, "manhattan": _manhattan, "jaccard": _jaccard}


def _text(value: Any, description: str | None = None):
    """What is asked about: a request's input as text, or a name on the map with its description."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return f"{text}: {description}" if description else text
