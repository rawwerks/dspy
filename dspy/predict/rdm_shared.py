"""
What `dspy.experimental.RDM` and its helpers share.

- How a decision is read: the options of a Choice, how likely a yes is, and which options of an
  output always stay on offer.
- State that is the program's. A map changes while a program runs: a function is written, a
  label is made. `Kept` holds such things as plain data and is a parameter of the program, so
  `save` writes it, `load` reads it, and a copy of the program has its own.
- A record of changes: `changed` puts one in the run's trajectory, on the request whose tool
  made it.
"""

import copy
from typing import Callable, get_args

from dspy.adapters.types.decision import Choice
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter

ACTIVE = "_rdm_call"  # where a running call is kept in DSPy's settings, so a call made inside it can join it
POINTS = "rdm_points"  # marks an output that points at one of the module's tools, once its options are filled in
ALWAYS = "rdm_always"  # on such an output, the names that stay on offer whatever a `fit` cuts


def options(output):
    """The options of a Choice output, each with its description; None for any other output."""
    kind = output.annotation
    if not (isinstance(kind, type) and issubclass(kind, Choice)) or kind.criteria() is None:
        return None
    criteria = kind.criteria()
    return [(value, criteria[str(value)]) for value in get_args(kind.model_fields["value"].annotation)]


def stays(output, option):
    """Whether an option is always offered: choosing none of them, and the names the module always offers, `finish` among them."""
    return option[0] is None or str(option[0]) in (output.json_schema_extra.get(ALWAYS) or ())


def likely(answer):
    """How likely a yes is, as the model gave it; a bare yes or no counts as certain."""
    probability = getattr(answer, "probability", None)
    return float(bool(getattr(answer, "value", answer))) if probability is None else float(probability)


class Kept(Parameter):
    """Plain data a module keeps, saved and loaded with the program it is part of."""

    def __init__(self, data: dict, loaded: Callable[[], None] | None = None):
        self.data = data
        self._loaded = loaded  # called once the data is what was loaded, to bring whatever is built from it up to date

    def dump_state(self, json_mode: bool = True) -> dict:
        return copy.deepcopy(self.data)

    def load_state(self, state: dict) -> None:
        self.data.clear()
        self.data.update(copy.deepcopy(state))
        if self._loaded is not None:
            self._loaded()

    def reset(self) -> None:
        """What was kept is not a demonstration to clear: it stays."""


def changed(what: dict) -> None:
    """Record a change to a map in the run's trajectory, under ``changes`` on the request whose tool made it.

    Outside a run there is no trajectory, and nothing is recorded.
    """
    active = settings.get(ACTIVE)
    if active is not None:
        with active.run.lock:
            active.run.trajectory[active.index].setdefault("changes", []).append(what)
