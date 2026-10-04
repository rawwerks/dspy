"""
Recursive Decision Model (RDM) module for DSPy.

An RLM lets a language model act on a state it cannot hold all at once, and call itself on
pieces of it. An RDM is the same idea for a model that returns decisions instead of text: a
System One model, which is given a state and typed questions and answers each with a decision
and a probability. A decision model cannot write, so it never makes a state. It decides, and
what it decides selects what makes the next state: code, a generative model, another module.

    call(request):                         a request is a state and typed questions
        answers <- predict(request)        asked once, and never again
        for each tool the answers select:  the tools run at once
            observation <- tool(...)       inside a tool, calling the module is a sub-call, one level down
            if an output pointed at it:    the same question again, about the state as it is now
        return answers, observations

An output typed `dspy.Tool` points: the module's tools are a map of names, each with a
description for the model and a thing for the code to do, and the map can change as it runs.

A run is a tree of such calls. Its shape is not written anywhere: it comes from what the
answers select and what the tools do. It stops where no answer selects a tool, where a
request comes round that was already asked on the way to it, or at a limit.

References:
    "Recursive Decision Models" (Weitekamp, 2026), https://raw.works/recursive-decision-models/
    "Recursive Language Models" (Zhang, Kraska, Khattab, 2025), https://arxiv.org/abs/2512.24601
"""

import contextvars
import copy
import dataclasses
import hashlib
import inspect
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from dspy.adapters.types.decision import Choice, Noul, Score
from dspy.adapters.types.tool import Tool
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict, serialize_object
from dspy.predict.rdm_fit import Fit
from dspy.predict.rdm_shared import ACTIVE as _ACTIVE
from dspy.predict.rdm_shared import ALWAYS as _ALWAYS
from dspy.predict.rdm_shared import POINTS as _POINTS
from dspy.predict.rdm_shared import options as _options
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.annotation import experimental
from dspy.utils.asyncify import asyncify
from dspy.utils.exceptions import format_error_for_lm
from dspy.utils.usage_tracker import UsageTracker

# Names a call takes for itself, and keys RDM attaches to every returned `Prediction`: no signature field may take them.
_RESERVED_NAMES = frozenset({"signature", "state", "max_depth", "observations", "termination_reason", "trajectory"})
# What can stop something other than a decision. When several did, the one that bounds the most is the reason.
_CUTS = ("max_tokens", "max_predictor_calls", "max_iters", "max_depth", "repeat")
_FINISH = "finish"  # the option that ends a turn-by-turn loop, as ReAct's `finish` does
_FINISH_DESC = "Stop: nothing more should be done."


class _UsageError(ValueError):
    """A module used in a way it cannot be: raised wherever it happens, a tool's sub-calls included."""


class Call(BaseModel):
    """The request that selected a tool. A tool receives it by taking a parameter annotated ``dspy.experimental.RDM.Call``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rdm: Any  # the module, for sub-calls from code that does not hold it
    tools: Any  # the module's tools in this run, the same dict: a tool put here is on offer from the run's next request
    prediction: Any  # the answers that selected the tool, each with its probabilities; None for `fit`
    inputs: dict  # the inputs of the request, whole
    state: Any  # what the run was given as `state=`, the same dict: what tools read and no request shows
    trajectory: list  # every request of the run so far
    depth: int
    iteration: int  # which turn of a pointer's loop this is
    requests_left: int

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # The module supplies it: a `dspy.Tool` has nothing to check it against.
        return {"description": "The request that selected the tool. Supplied by the module."}


@dataclass
class _Budget:
    """One module's limits on requests and tokens, and what has been used under it in a run."""

    max_predictor_calls: int
    max_tokens: int | None
    predictor_calls: int = 0
    tokens: int = 0

    def spent(self):
        """The limit that is used up, if one is."""
        if self.predictor_calls >= self.max_predictor_calls:
            return "max_predictor_calls"
        return "max_tokens" if self.max_tokens is not None and self.tokens >= self.max_tokens else None


@dataclass
class _Run:
    """One run: the history of every request made for it."""

    trajectory: list = field(default_factory=list)
    tracker: UsageTracker | None = None  # whatever was tracking usage around the run
    tools: dict = field(default_factory=dict)  # each module's tools for this run, by the module's id
    offered: dict = field(default_factory=dict)  # the names each of a module's sources has on its map in this run
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class _Active:
    """The request whose tool is running: what a call made from inside that tool is a sub-call of."""

    run: _Run
    depth: int
    index: int
    limit: int  # the depth at which a call made from here is answered and its tools are not run
    path: dict  # every request asked on the way here, each with the answers it got
    module: Any = None  # the module whose tool is running
    budgets: tuple = ()  # the budget of every module the request is under: the run's first
    state: dict = field(default_factory=dict)  # what tools read and no request shows: the nearest call's that gave one
    tool: str | None = None  # the tool that is running
    cuts: list = field(default_factory=list)  # what stopped something in a call made from here, other than a decision


@experimental
class RDM(Module):
    """Recursive Decision Model module.

    Answers a signature's questions once, as ``dspy.Predict`` does, and lets the answers select
    tools. An output named after a tool calls it on a yes or a chosen option. An output typed
    ``dspy.Tool`` points at one of the module's tools, as ``dspy.ReAct``'s ``next_tool_name`` does
    for a language model: each request offers the tools as they are then, by name and description,
    with ``finish``, and after the tool pointed at has run the same question is put again, showing
    the inputs as they are now and, in an input named ``observation``, what the tool returned. A
    run works on its own copy of ``self.tools``: a tool put in ``call.tools`` is on offer from the
    run's next request, and one put in ``self.tools`` from the next run.

    A tool can be a function, a generative module, or another ``RDM``. Inside a tool, calling this
    module is a sub-call: one level deeper, in the same run, with its prediction returned to the
    tool. What a tool returns is its observation. Nothing here generates text, so it runs on a
    System One client such as ``dspy.experimental.TypeSafe`` as well as on a generative LM.

    Args:
        signature: The questions to answer. An output named after a tool calls that tool, and an
            output typed ``dspy.Tool`` points at any of them.
        tools: Functions, callable objects, modules or ``dspy.Tool`` instances the answers can select.
            A tool's parameters are filled by name from the request's answers, then its inputs, and a
            parameter annotated ``dspy.experimental.RDM.Call`` receives the request itself (its prediction with
            probabilities, the run's ``state``, the trajectory, the module). Something that offers
            tools of its own, such as a ``dspy.experimental.RDM.Repl``, is given whole: each
            request's map holds what it offers then, and it is saved and copied with the module.
        always: Names of tools that stay on offer whatever a ``fit`` cuts from the map, as
            ``finish`` does: a tool that writes, say.
        fit: Called with each request before it is asked, as ``fit(signature, inputs, call)``, and
            returns the signature and inputs to ask in their place. It can make sub-calls through
            ``call.rdm``, as a tool can. ``dspy.experimental.RDM.Fit()`` cuts down what one request cannot hold.
            None asks every request as it is.
            Several, given as a list, are applied in order.
        finish: Whether an output typed ``dspy.Tool`` is offered ``finish`` beside the tools. Without
            it a loop ends at a limit, at a repeat, or at a tool named ``finish`` if there is one:
            for a game, say, where stopping is not the model's to choose.
        max_iters: Turns one pointer's loop may take.
        max_depth: The depth at which a call's tools are not run. The first call is at depth 0, so
            1 lets it use its tools and makes every sub-call a leaf, and 0 makes it a leaf itself.
            It is counted from where the module was called, so a module called inside another's
            tool goes as deep as it would alone.
        max_predictor_calls: Requests answered in one run, sub-calls included. A module called
            inside another's tool joins that run: the run's limits hold for it, and so do its own,
            for what it does there.
        max_tokens: Tokens, input and output, one run may use in total, as the clients report
            them: those of its requests and those its tools use. None for no limit.

    Examples:
        ```python
        from functools import partial

        import dspy
        from dspy.experimental import RDM, TypeSafe

        house = {"tv": "off", "lights": "on"}

        def switch(device, to):
            house[device] = to
            return f"{device}: now {to}"

        class Operate(dspy.Signature):
            \"\"\"Operate a smart home, one step at a time.\"\"\"

            task: str = dspy.InputField()
            house: dict = dspy.InputField()
            observation: str = dspy.InputField(desc="What the last step did.")
            next: dspy.Tool = dspy.OutputField(desc="What should happen next for `inputs.task`?")

        steps = [
            dspy.Tool(partial(switch, "tv", "on"), name="tv on", desc="Turn the TV on."),
            dspy.Tool(partial(switch, "lights", "off"), name="lights off", desc="Turn the lights off."),
        ]
        rdm = RDM(Operate, tools=steps)
        rdm.set_lm(TypeSafe("jev-latest"))  # the decision model points at every turn
        result = rdm(task="Movie time.", house=house)
        print(result.observations["next"])  # [['tv on', 'tv: now on'], ['lights off', 'lights: now off']]
        ```
    """

    Call = Call
    Fit = Fit

    def __init__(
        self,
        signature: type[Signature] | str,
        tools: list[Callable | Tool] | None = None,
        fit: Callable | list[Callable] | None = None,
        always: list[str] | None = None,
        finish: bool = True,
        max_iters: int = 20,
        max_depth: int = 3,
        max_predictor_calls: int = 256,
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.signature = ensure_signature(signature)
        reserved = _RESERVED_NAMES.intersection(self.signature.fields)
        if reserved:
            raise ValueError(f"Field name(s) {sorted(reserved)} are reserved by RDM. Rename these fields.")
        if max_iters < 1 or max_predictor_calls < 1:
            raise ValueError("max_iters and max_predictor_calls must be at least 1: the first request counts.")
        if isinstance(tools, dict):
            raise TypeError(
                "tools must be a list, not a dict. Change tools={'name': func} to tools=[func] "
                "(tool names are inferred from function names, or use dspy.Tool(func, name='custom_name'))"
            )
        self.tools = {}
        self.sources = []  # what offers tools of its own: its tools are on each request's map as they are then
        self.always = [str(name) for name in always or []]
        for tool in tools or []:
            if not isinstance(tool, Tool) and hasattr(tool, "offers"):
                self.sources.append(tool)
                continue
            tool = tool if isinstance(tool, Tool) else Tool(tool)
            if tool.name in self.tools:
                raise ValueError(f"Duplicate tool name '{tool.name}'")
            self.tools[tool.name] = tool
        self.predict = Predict(self.signature)
        self.fit = fit
        self.finish = finish
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.max_predictor_calls = max_predictor_calls
        self.max_tokens = max_tokens

    def forward(
        self, signature: type[Signature] | str | None = None, state: dict | None = None, max_depth: int | None = None, **inputs
    ) -> Prediction:
        """Answer a signature's questions about the inputs, and run the tools the answers select.

        Called from inside one of its own tools, this is a sub-call: it joins the run in progress,
        one level deeper, and counts against the same limits.

        Args:
            signature: Questions to put in place of the module's own, for this call.
            state: Anything the tools of this call, and of the calls under it, should be able to
                read that no request should show, read through ``dspy.experimental.RDM.Call``.
            max_depth: How many levels below this call may still run tools, when that is fewer
                than the run allows. 0 asks for the answers only: no tool is run for it.
            **inputs: Input values matching the signature's input fields.

        Returns:
            Prediction with the answers; ``observations``, what each selected tool returned, and
            for an output typed as a tool, every tool it pointed at in turn, each with what it
            returned; ``termination_reason`` (``final``; or what stopped something in this call or
            below it other than a decision: ``repeat`` for a request that had already been asked
            on the way to it, or the limit ``max_depth``, ``max_iters``, ``max_predictor_calls`` or
            ``max_tokens``); and, for the first call of a run, ``trajectory``: every request of
            the run in the order it was made, each with its ``depth``, its ``iteration``, the
            ``parent`` request and ``tool_name`` it came from, the ``inputs`` it showed, the
            ``outputs`` it got, the ``model`` that gave them, the ``tokens`` it used, its
            ``observations``, ``tool_tokens`` for a tool that used a model of its own
            (``fit_tokens`` for ``fit``), and ``unasked`` for outputs that had one option and were
            answered without asking.

        Raises:
            ValueError: If inputs do not match the signature, or a tool cannot be given its arguments.
        """
        request = {**inputs, "signature": signature, "max_depth": max_depth}
        active = settings.get(_ACTIVE)
        if active is not None:
            # The sub-call: this module called from inside a tool that is running. State it is given
            # is its own and that of the calls under it, so a module works the same wherever it is called.
            return self._children(active, [request], state=state)[0]
        run = _Run(tracker=settings.usage_tracker)
        asked, inputs, limit, answers_only = self._request(run, request, 0, self.max_depth)
        budgets = (_Budget(self.max_predictor_calls, self.max_tokens),)
        with run.lock:
            opened = self._open(run, asked, inputs, 0, None, None, 0, {}, _key(self, asked, inputs), budgets)
        prediction = self._turns(run, opened, asked, inputs, 0, limit, {}, budgets, dict(state or {}), answers_only)
        return Prediction(**dict(prediction.items()), trajectory=run.trajectory)

    async def aforward(
        self, signature: type[Signature] | str | None = None, state: dict | None = None, max_depth: int | None = None, **inputs
    ) -> Prediction:
        """Async version of forward(). The calls are synchronous and run in a worker thread, where an async tool is awaited."""

        def forward():
            with settings.context(allow_tool_async_sync_conversion=True):
                return self.forward(signature=signature, state=state, max_depth=max_depth, **inputs)

        return await asyncify(forward)()

    def batch(self, examples, *args, **kwargs):
        """As ``Module.batch``, taking ``dspy.Example``s or plain dicts of keyword arguments.

        Inside a run the examples are sub-calls: admitted in the order given, and asked at once.
        A dict may name a ``signature`` and a ``max_depth`` beside the inputs, as a call may.
        """
        examples = [example if isinstance(example, Example) else Example(**example).with_inputs(*example) for example in examples]
        active = settings.get(_ACTIVE)
        if active is None:
            return super().batch(examples, *args, **kwargs)
        return self._children(active, [dict(example.inputs()) for example in examples])

    def _tools(self, run):
        """This module's tools for a run: a copy of ``self.tools`` made when the run first needs it.

        A tool can add to or remove from it through ``call.tools``, and the next request of the
        run offers the tools as they are then. Other runs, on other threads, are not affected.
        """
        with run.lock:
            tools = run.tools.setdefault(id(self), dict(self.tools))
            # What a source offers is on the map as it is now: a function written a moment ago, and not one removed.
            for source in self.sources:
                offered = {tool.name: tool for tool in source.offers}
                mine = run.offered.setdefault((id(self), id(source)), set())
                for name in mine - set(offered):
                    tools.pop(name, None)
                taken = [name for name in offered if name in tools and name not in mine]
                if taken:
                    raise _UsageError(f"{taken} are offered by `{type(source).__name__}` and are already names on the map.")
                tools.update(offered)
                mine.clear()
                mine.update(offered)
            return tools

    def _request(self, run, request, depth, limit):
        """A request as its signature, its inputs, the depth at which its tools stop, and whether only its answers are wanted."""
        signature = self._offered(run, ensure_signature(request.get("signature") or self.signature))
        inputs = {key: value for key, value in request.items() if key not in ("signature", "max_depth")}
        pointers = _pointers(signature)
        if len(pointers) > 1:
            raise _UsageError(f"Outputs {pointers} are typed `dspy.Tool`; a signature may have one output that points at a tool.")
        if pointers and "observation" in signature.input_fields:
            # Before the first tool has run there is nothing to observe.
            inputs.setdefault("observation", "")
        if set(inputs) != set(signature.input_fields) or _RESERVED_NAMES.intersection(signature.fields):
            raise _UsageError(
                f"The call has inputs {sorted(inputs)}; its signature takes {sorted(signature.input_fields)}, "
                f"and no field may be named any of {sorted(_RESERVED_NAMES)}."
            )
        further = request.get("max_depth")
        # A call keeps to the limit of the run it is in, and to its own where that is lower.
        return signature, inputs, min(limit, depth + (self.max_depth if further is None else further)), further == 0

    def _offered(self, run, signature):
        """The signature with its output typed as a tool offering the run's tools as they are now, and `finish`."""
        for name in _pointers(signature):
            tools = self._tools(run)
            options = [(key, _tool(tools, key).desc or "") for key in list(tools) if key != _FINISH]
            if _FINISH in tools or self.finish:
                options.append((_FINISH, _tool(tools, _FINISH).desc if _FINISH in tools else _FINISH_DESC))
            signature = signature.with_updated_fields(name, type_=Choice[tuple(options)], **{_POINTS: True, _ALWAYS: [_FINISH, *self.always]})
        return signature

    def _children(self, active, requests, state=None):
        """Ask requests one level below the request whose tool is running, at once. Returns their predictions in order."""
        run, depth = active.run, active.depth + 1
        # Another module called inside a tool joins the run, and behaves there as it does alone: its
        # own limits hold for what it does, its depth is counted from where it was called, and the
        # run's requests and tokens are counted for it too.
        own = active.module is self
        budgets = active.budgets if own else (*active.budgets, _Budget(self.max_predictor_calls, self.max_tokens))
        state = active.state if state is None else dict(state)
        asked = [self._request(run, request, depth, active.limit if own else math.inf) for request in requests]
        keys = [_key(self, signature, inputs) for signature, inputs, _, _ in asked]
        with run.lock:
            # They are admitted in the order given, so which ones a limit stops does not depend on timing.
            opened = [
                self._open(run, signature, inputs, depth, active.index, active.tool, 0, active.path, key, budgets)
                for (signature, inputs, _, _), key in zip(asked, keys, strict=True)
            ]
        calls = [
            lambda entry=entry, request=request: self._turns(
                run, entry, *request[:2], depth, request[2], active.path, budgets, state, request[3]
            )
            for entry, request in zip(opened, asked, strict=True)
        ]
        predictions = _in_parallel(calls)
        # What stopped something in a call below, other than a decision, stopped it for the call above too.
        active.cuts.extend(prediction.termination_reason for prediction in predictions if prediction.termination_reason != "final")
        return predictions

    def _turns(self, run, opened, signature, inputs, depth, limit, path, budgets, state, answers_only):
        """Answer a request that has been recorded and, while its output typed as a tool points at a tool, put it again.

        `limit` is the depth at which tools are no longer run, and `path` the requests asked on the way here.

        Each turn after the first shows the inputs as they are now, so a tool that changed an
        object the request shows changed what the next turn shows, and an input named
        `observation` shows what the last tool returned. The turns are a loop at one level, so a
        loop may be as long as `max_iters` allows. Returns the first turn's answers, with every
        tool pointed at in turn, each with what it returned.
        """
        pointer = next(iter(_pointers(signature)), None)
        # Each turn's options are made from the first turn's signature, so turns do not stack up copies of it.
        first, done, reasons, iteration, base = None, [], [], 0, signature
        while True:
            prediction, step = self._answer(run, *opened, signature, inputs, depth, limit, path, budgets, state, iteration, answers_only)
            first = prediction if first is None else first
            reasons.append(prediction.termination_reason)
            if step is None:
                break
            name, result, path = step
            done.append([name, result])
            if name == _FINISH:
                break
            observed = {"observation": _observed(result)} if "observation" in signature.input_fields else {}
            signature, inputs, limit, answers_only = self._request(run, {"signature": base, **inputs, **observed}, depth, limit)
            iteration, parent = iteration + 1, opened[0]
            if iteration >= self.max_iters:
                stopped = {
                    "depth": depth, "iteration": iteration, "parent": parent, "tool_name": name,
                    "inputs": serialize_object(inputs), "skipped": "max_iters",
                }
                with run.lock:
                    run.trajectory.append(stopped)
                reasons.append("max_iters")
                break
            key = _key(self, signature, inputs)
            with run.lock:
                opened = self._open(run, signature, inputs, depth, parent, name, iteration, path, key, budgets)
        observations = {**first.observations, **({pointer: done} if pointer else {})}
        reason = min((reason for reason in reasons if reason != "final"), key=_CUTS.index, default="final")
        answers = {name: value for name, value in first.items() if name not in ("observations", "termination_reason")}
        return Prediction(**answers, observations=observations, termination_reason=reason)

    def _open(self, run, signature, inputs, depth, parent, tool_name, iteration, path, key, budgets):
        """Record a request and count it against the limits. The caller holds the run's lock."""
        # What the request showed, as it showed it: a tool may change an input object later.
        shown = serialize_object(inputs)
        entry = {"depth": depth, "iteration": iteration, "parent": parent, "tool_name": tool_name, "inputs": shown}
        run.trajectory.append(entry)
        # A Choice with one option leaves nothing to decide: it is answered with that option and
        # not asked. A request made only of such questions is not sent, and counts against no limit.
        only = {name: _only(output) for name, output in signature.output_fields.items()}
        only = {name: answer for name, answer in only.items() if answer is not None}
        if key in path:
            # The same questions about the same state, asked on the way here, would get the same
            # answers and select the same tools again. It is not asked: the run has come round.
            entry["repeat"] = True
        elif len(only) < len(signature.output_fields):
            _admit(budgets, entry)
        return len(run.trajectory) - 1, entry, only, key

    def _answer(self, run, index, entry, only, key, signature, inputs, depth, limit, path, budgets, state, iteration, answers_only):
        """Ask a request that has been recorded, and run the tools its answers select.

        Returns the prediction and, when its output typed as a tool pointed at a tool or at
        `finish`, that tool's name, what it returned, and the requests asked on the way.
        """
        if "skipped" in entry:
            return _unanswered(signature, entry["skipped"]), None
        if "repeat" in entry:
            entry["outputs"] = {name: serialize_object(answer) for name, answer in path[key].items()}
            return Prediction(**path[key], observations={}, termination_reason="repeat"), None
        active = _Active(run, depth, index, limit, dict(path), module=self, budgets=budgets, state=state)
        given = inputs
        asked = signature
        for name in only:
            asked = asked.delete(name)
        answered = {}
        if asked.output_fields:
            if self.fit is not None:
                # A request that is more than one request can hold is cut down first, by whatever
                # `fit` asks. What it asks are sub-calls of this one.
                with _usage(run) as usage, settings.context(**{_ACTIVE: active}):
                    for fit in self.fit if isinstance(self.fit, (list, tuple)) else [self.fit]:
                        asked, inputs = fit(asked, inputs, self._context(run, active, iteration, given, None))
                entry["inputs"] = serialize_object(inputs)
                if _tokens(usage):
                    # What `fit` used on a model of its own. Its sub-calls are counted on their own requests.
                    entry["fit_tokens"] = _tokens(usage)
                    _use(run, budgets, entry["fit_tokens"])
            prediction = self._ask(run, entry, asked, inputs, budgets)
            if prediction is None:
                return _unanswered(signature, entry["skipped"]), None
            answered = {name: prediction[name] for name in asked.output_fields}
        if only:
            entry["tokens"] = entry.get("tokens", 0)
            entry["outputs"] = {name: serialize_object({**answered, **only}[name]) for name in signature.output_fields}
            entry["unasked"] = sorted(only)
        prediction = Prediction(**{name: {**answered, **only}[name] for name in signature.output_fields})
        entry["observations"] = observations = {}
        # What is asked from here on, below this request or after it, has this request on its way.
        active.path[key] = dict(prediction.items())

        # An output named after a tool calls it: a yes calls it, and so does a chosen option. An
        # output typed as a tool calls the tool it points at. Each is (selecting output, tool).
        tools = self._tools(run)
        pointers = _pointers(signature)
        selected = [(name, name) for name in signature.output_fields if name in tools and name not in pointers and _selects(prediction[name])]
        selected += [(name, _value(prediction[name])) for name in pointers if _value(prediction[name]) in tools]
        finished = [name for name in pointers if _value(prediction[name]) == _FINISH and _FINISH not in tools]
        if (selected or finished) and depth >= limit:
            # The leaf: at max_depth a call is answered, and what it selected is not run. When the
            # caller asked for the answers only, nothing was kept from being done.
            selected, finished = [], []
            active.cuts.extend(() if answers_only else ("max_depth",))
        call = self._context(run, active, iteration, given, prediction)
        bound = [(selector, name, _tool(tools, name)) for selector, name in selected]
        bound = [(selector, name, tool, self._bind(tool, prediction, signature, given, call)) for selector, name, tool in bound]
        calls = [
            lambda name=name, tool=tool, arguments=arguments: self._run_tool(run, dataclasses.replace(active, tool=name), entry, name, tool, arguments)
            for _, name, tool, arguments in bound
        ]
        step = (_FINISH, None, active.path) if finished else None
        for (selector, name, _, _), result in zip(bound, _in_parallel(calls), strict=True):
            observations[selector] = result
            if selector in pointers:
                step = (name, result, active.path)
        # A limit that stopped something below this call stopped it for this call.
        reason = min(active.cuts, key=_CUTS.index, default="final")
        return Prediction(**dict(prediction.items()), observations=observations, termination_reason=reason), step

    def _ask(self, run, entry, signature, inputs, budgets):
        """Put one admitted request to the model and record what it answered. Returns None for one that was not asked."""
        with run.lock:
            if any(budget.max_tokens is not None and budget.tokens >= budget.max_tokens for budget in budgets):
                # What was asked before it has used the tokens since it was admitted.
                entry["skipped"] = "max_tokens"
                for budget in budgets:
                    budget.predictor_calls -= 1
                return None
        predictor = self._predictor(signature)
        with _usage(run) as usage:
            prediction = predictor(signature=signature, **inputs)
        entry["tokens"] = _tokens(usage)
        # Which model answered: a run may put its requests to more than one.
        entry["model"] = getattr(predictor.lm or settings.lm, "model", None)
        _use(run, budgets, entry["tokens"])
        entry["outputs"] = {name: serialize_object(prediction[name]) for name in signature.output_fields}
        return prediction

    def _run_tool(self, run, active, entry, name, tool, arguments):
        """Run one tool and return what it returned. A call of the module inside it is a sub-call of `active`."""
        try:
            # A tool's own use of a model, a generative one say, is counted with the run's. What
            # the requests it asks through the module use is counted on those requests.
            with _usage(run) as usage, settings.context(**{_ACTIVE: active}):
                # Through `dspy.Tool`, as ReAct calls one: its callbacks fire, and an async tool is handled as DSPy handles it.
                return tool(**arguments)
        except _UsageError:
            # A module used wrongly inside a tool is the caller's error wherever it happens.
            raise
        except Exception as error:
            # As in ReAct, a tool that fails is an observation and not the end of the run.
            return f"Execution error in {name}: {format_error_for_lm(error, traceback_frames=5)}"
        finally:
            used = _tokens(usage)
            if used:
                _use(run, active.budgets, used)
                with run.lock:
                    entry.setdefault("tool_tokens", {})[name] = used

    def _context(self, run, active, iteration, inputs, prediction):
        """The request, as a tool or `fit` receives it."""
        with run.lock:
            left, trajectory = min(budget.max_predictor_calls - budget.predictor_calls for budget in active.budgets), list(run.trajectory)
        return Call(
            rdm=self, tools=self._tools(run), prediction=prediction, inputs=inputs, state=active.state, trajectory=trajectory,
            depth=active.depth, iteration=iteration, requests_left=left,
        )

    def _predictor(self, signature):
        """The predictor for a request's questions.

        A predictor the program holds that declares these questions answers them: the module's
        own, `fit`'s, or one given to the module as an attribute for the questions of a sub-call
        (``rdm.judge = dspy.Predict(Judge)``). Its parameters, such as thresholds, are then the
        program's: saved with it, and fitted by an optimizer such as ReAnchor.

        Otherwise parameters set on `predict` apply by output name. Predict refuses a signature
        that lacks an output it holds parameters for, so such a request is answered by a copy
        that keeps the parameters of the outputs it does ask.
        """
        outputs = set(signature.output_fields)
        if not outputs <= set(self.predict.signature.output_fields):
            for _, predictor in self.named_predictors():
                if predictor is not self.predict and outputs <= set(predictor.signature.output_fields) and set(predictor.fields) <= outputs:
                    return predictor
        declared = self.predict.signature.output_fields

        def applies(name):
            # Parameters fitted to one answer space, a Choice's weights say, are not another's: a
            # request that asks an output of the same name over other options is answered without them.
            return name in signature.output_fields and (name not in declared or declared[name].annotation == signature.output_fields[name].annotation)

        fields = {name: config for name, config in self.predict.fields.items() if applies(name)}
        # The module's own predictor answers only when every parameter it holds is for an output
        # it declares and this request asks. Predict checks the two against each other.
        if fields == self.predict.fields and all(name in declared for name in fields):
            return self.predict
        predictor = copy.copy(self.predict)
        predictor.signature, predictor.fields = signature, fields
        return predictor

    def _bind(self, tool, prediction, signature, inputs, call):
        """A tool's arguments, every one taken from what already exists.

        A parameter annotated ``dspy.experimental.RDM.Call`` receives the request. Any other is filled by name
        from the request's answers, then from its inputs. A required parameter that names none of
        these is refused, so a misspelt name cannot quietly receive something else.
        """
        if isinstance(tool.func, Module):
            # A module is called through its signature, which says what it takes.
            parameters = dict.fromkeys(_signature_of(tool.func, tool.name).input_fields, (True, None))
        else:
            parameters = _parameters(tool)
        arguments, missing = {}, []
        for parameter, (required, annotation) in parameters.items():
            if annotation is Call:
                arguments[parameter] = call
            elif parameter in signature.output_fields:
                arguments[parameter] = _value(prediction[parameter])
            elif parameter in inputs:
                arguments[parameter] = inputs[parameter]
            elif required:
                missing.append(parameter)
        if missing:
            raise _UsageError(
                f"Tool '{tool.name}' takes {missing}, which name no answer or input of the request. A parameter is filled "
                f"by name from {sorted({*signature.output_fields, *inputs})}, and one annotated `dspy.experimental.RDM.Call` "
                "receives the request."
            )
        return arguments


def _unanswered(signature, reason):
    """The prediction of a request a limit kept from being asked: every output None."""
    return Prediction(**dict.fromkeys(signature.output_fields), observations={}, termination_reason=reason)


def _admit(budgets, entry):
    """Count a request against the limits of every module it is under, or mark it as one a limit stopped. The caller holds the run's lock."""
    # A request past a limit is recorded and not answered, so nothing is dropped silently.
    spent = next((budget.spent() for budget in budgets if budget.spent()), None)
    if spent:
        entry["skipped"] = spent
    else:
        for budget in budgets:
            budget.predictor_calls += 1


def _use(run, budgets, tokens):
    """Count tokens against the limits of every module they were used under."""
    with run.lock:
        for budget in budgets:
            budget.tokens += tokens


def _key(module, signature, inputs):
    """What makes two requests the same request: the same module's, with the same questions about the same state."""
    questions = [(name, str(asked.annotation), asked.json_schema_extra.get("desc")) for name, asked in signature.fields.items()]
    whole = json.dumps([id(module), signature.instructions, questions, inputs], default=str)
    return hashlib.sha256(whole.encode()).hexdigest()


def _pointers(signature):
    """The outputs that point at one of the module's tools: typed as a tool, or offering the tools as options."""
    return [name for name, output in signature.output_fields.items() if output.annotation is Tool or output.json_schema_extra.get(_POINTS)]


def _parameters(tool):
    """A tool's parameters, each with whether it must be given and its annotation. A `functools.partial` takes only what it leaves open."""
    try:
        try:
            parameters = inspect.signature(tool.func, eval_str=True).parameters.values()
        except NameError:  # an annotation written as a string that cannot be resolved
            parameters = inspect.signature(tool.func).parameters.values()
    except (TypeError, ValueError):
        return {arg: ("default" not in schema, None) for arg, schema in (tool.args or {}).items()}
    takes = {p.name: (p.default is p.empty, p.annotation) for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        # A function that takes whatever it is given says what it takes on the tool, as a tool made
        # from an MCP server does. Each is filled where something has its name, and left out otherwise.
        open_ended = {p.name for p in parameters if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        takes.update({arg: (False, None) for arg in tool.args or {} if arg not in takes and arg not in open_ended})
    return takes


def _observed(result):
    """What a tool returned, as an input shows it: a text as it is, anything else as JSON, as inputs are shown."""
    if result is None or isinstance(result, str):
        return result or ""
    return json.dumps(serialize_object(result), default=str, ensure_ascii=False)


def _tool(tools, name):
    """The tool a map holds under a name, as a ``dspy.Tool`` of that name: a plain function is wrapped."""
    tool = tools[name]
    return tool if isinstance(tool, Tool) else Tool(tool, name=name)


def _value(answer: Any):
    """An answer as a plain value: what a decision was, without its probabilities."""
    return answer.value if isinstance(answer, (Noul, Choice, Score)) else answer


def _selects(answer: Any):
    """Whether an answer calls the tool it is named after: a yes does, and so does any chosen option but None."""
    value = _value(answer)  # a Score's value is a number, which calls nothing
    return value is not None and value is not False and value != "" and not isinstance(value, float)


def _only(output):
    """The answer to a Choice that has one option: that option, as certain. None for any other output."""
    offered = _options(output)
    if offered is None or len(offered) != 1:
        return None
    value = offered[0][0]
    return output.annotation(value=value, confidence=1.0, probabilities={str(value): 1.0})


@contextmanager
def _usage(run):
    """Track what is used inside, apart from any tracker around it, and pass it on to whatever tracks the run.

    DSPy's trackers nest, so a request asked from inside a tool would be counted on the request
    and again by the tool. Each request and each tool counts only what it used itself.
    """
    tracker = UsageTracker()
    try:
        with settings.context(usage_tracker=tracker):
            yield tracker
    finally:
        if run.tracker is not None:
            with run.lock:
                for model, entries in tracker.usage_data.items():
                    run.tracker.usage_data[model].extend(entries)


def _tokens(usage):
    """The tokens a usage tracker recorded, input and output, as the client reported them. A cached request reports none."""
    used = [entry for entries in usage.usage_data.values() for entry in entries]
    return sum(entry.get("total_tokens") or (entry.get("prompt_tokens") or 0) + (entry.get("completion_tokens") or 0) for entry in used)


def _signature_of(module, name):
    signature = getattr(module, "signature", None)
    if signature is None:
        predictors = module.predictors()
        signature = predictors[0].signature if len(predictors) == 1 else None
    if signature is None:
        raise _UsageError(f"Tool '{name}' is a module with no signature to take its parameters from.")
    return signature


def _in_parallel(calls):
    """Run calls at once and return their results in order. Each thread sees the caller's DSPy settings."""
    if len(calls) <= 1:
        return [call() for call in calls]
    with ThreadPoolExecutor(max_workers=min(len(calls), settings.num_threads)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, call) for call in calls]
        return [future.result() for future in futures]
