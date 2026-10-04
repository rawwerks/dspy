"""
Filter: sorts what a decision model rated, likeliest first, and keeps the likeliest.

A tool of a `dspy.experimental.RDM` often has to act on more than the one answer a decision came to: follow
the two likeliest options of a `Choice`, or keep the items a yes or no was said of. `Filter` is
the sorting and the keeping:

    likeliest = RDM.Filter(top=2)

    def category(listing, call: RDM.Call):
        for option, probability in likeliest(call.prediction.category):   # the Choice's options, likeliest first
            ...
        verdicts = call.rdm.batch([{"signature": Fits, "leaf": leaf, "max_depth": 0} for leaf in leaves])
        return likeliest(zip(leaves, (verdict.fits for verdict in verdicts)))  # the leaves said yes of, likeliest first

It has no cut of its own. A yes is a yes where its predictor's threshold says so, and that
threshold is a parameter of the program, which ReAnchor fits.
"""

from typing import Any, get_args

from dspy.adapters.types.decision import Choice, Noul
from dspy.predict.rdm_shared import likely
from dspy.primitives.prediction import Prediction
from dspy.utils.annotation import experimental


@experimental
class Filter:
    """Sorts what a decision model rated by its probabilities, and keeps the likeliest.

    Called with what was rated, it returns ``(thing, probability)`` pairs, likeliest first; things
    rated alike keep the order they came in. What was rated can be:

    - a ``Choice`` answer: its options, each as the value it was declared with (or, when it came
      with no probabilities, the one value chosen, as certain);
    - a prediction: its yes or no answers, by the name of the output;
    - a mapping or pairs of anything with the yes or no given about it, or with a probability.

    Of yes or no answers only the yeses are kept. Whether an answer is a yes is its predictor's
    threshold, so the cut is a parameter of the program and not of the filter: set it on the
    predictor (``rdm.judge.fields["fits"] = {"threshold": 0.7}``) or let ReAnchor fit it.

    Args:
        top: How many to keep at most. None keeps all.

    Examples:
        ```python
        likeliest = RDM.Filter(top=2)

        def go(call: RDM.Call):
            # Follow the two ways the model rated likeliest, not only the one it chose.
            return [call.rdm(signature=Look, way=way).there.value for way, probability in likeliest(call.prediction.go)]
        ```
    """

    def __init__(self, top: int | None = None):
        if top is not None and top < 1:
            raise ValueError("top must be at least 1: a filter that keeps nothing has nothing to give.")
        self.top = top

    def __call__(self, rated) -> list[tuple[Any, float]]:
        """The things rated, each with its probability, likeliest first.

        Args:
            rated: A ``Choice`` answer, a prediction, or a mapping or pairs of things with their answers.
        """
        kept = [(thing, _probability(answer)) for thing, answer in _pairs(rated) if not (isinstance(answer, Noul) and not answer.value)]
        return sorted(kept, key=lambda pair: -pair[1])[: self.top]


def _pairs(rated):
    """What was rated as pairs of a thing and the answer or probability given about it."""
    if isinstance(rated, Choice):
        values = {str(value): value for value in get_args(type(rated).model_fields["value"].annotation)}
        return [(values.get(label, label), probability) for label, probability in rated.probabilities.items()]
    if isinstance(rated, Prediction):
        return [(name, answer) for name, answer in rated.items() if isinstance(answer, Noul)]
    return list(rated.items()) if isinstance(rated, dict) else list(rated)


def _probability(answer):
    """How likely an answer is: a yes or no's probability of yes, or the number given."""
    if isinstance(answer, Noul):
        return likely(answer)
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        raise TypeError(f"{answer!r} cannot be read as a probability: give a yes or no answer, or a number.")
    return float(answer)
