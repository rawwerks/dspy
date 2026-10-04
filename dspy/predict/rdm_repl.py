"""
Repl: a Python REPL where what a language model writes can be pointed at by a decision model.

A decision model can only point, so a `dspy.experimental.RDM` can only do what is already on
its map. A language model can write. `Repl` is the environment the two share: a sandboxed
interpreter, as `dspy.RLM` runs in, where the tools it is given are functions.

    repl = RDM.Repl(tools=[do, read])                     # the sandbox, and the functions that exist in it
    writer = dspy.RLM(Write, tools=[read, repl.define])   # a language model that writes by calling `define`
    rdm = RDM(Operate, tools=[*steps, repl, dspy.Tool(writer, name="write")], always=["write"])

    the decision model points at `write`         nothing on the map does what is still needed
    the writer calls `define(name, ...)`         with the source of one function
    the function is defined in the REPL          and its name is on the map from the next request
    the decision model points at the name        and the REPL runs it

`repl.define` and `repl.remove` are plain functions, so a language model writes as it does
anything else: as the tools of a `dspy.RLM`, or through any function of yours that calls them.
The language model never chooses what happens. It makes new things that can be pointed at,
with what a decision model cannot do: loops, arithmetic, conditions.

A written function is a tool like any other: its parameters are filled by name from the request's
answers, then its inputs. So what a function should be told (how many, which one) is a question
of the request, answered by the decision model, and not a number written into the function.

Something found out and worth keeping is written the same way, as a function that returns it:
`define("boiler", "What was found wrong with the boiler.", "def boiler():\n    return 'The pilot light is out.'")`.
Pointing at it reads it. The REPL is the one place a program keeps what it made, as an RLM's is.

What was written is `repl.written`, plain data that is part of the program: `save` writes it,
`load` defines it again, and a copy of the program has a REPL of its own with the same functions.

So an RDM runs in one of three ways:

- with no REPL, the map holds only what was written ahead of time;
- with one REPL, what is written stays for the rest of the run and the runs after it;
- with several, each from `repl.spawn(...)`, every environment is isolated from the others: a
  function written in one does not exist in another, and a name on a map can stand for a whole
  environment, so that pointing at it routes a request there.

The interpreter comes from `interpreter_factory`, as for `dspy.RLM`: `dspy.PythonInterpreter`
(Deno) by default, or what `dspy.configure(interpreter_factory=...)` names, such as Monty.
"""

import ast
import contextvars
import copy
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from dspy.adapters.types.tool import Tool
from dspy.dsp.utils.settings import settings
from dspy.predict.rdm_shared import ACTIVE, Kept, changed
from dspy.predict.rlm import RLM, _strip_code_fences
from dspy.primitives.code_interpreter import CodeExecutionError, _create_interpreter
from dspy.primitives.module import Module
from dspy.utils.annotation import experimental

_ARGUMENTS = "_rdm_arguments"  # the variable a written function's arguments are given to the REPL in
_RUNNING = contextvars.ContextVar("rdm_repl_running", default=frozenset())  # the REPLs that are running code, on the way to here


@experimental
class Repl(Module):
    """A Python REPL whose functions a ``dspy.experimental.RDM`` can point at.

    Given to an RDM among its ``tools``, it offers every function written so far: each request's
    map holds them as they are then. ``define(name, description, code)`` writes one, and
    ``remove(name)`` takes one away; both are plain functions, to give to a ``dspy.RLM`` as
    tools or to call from a tool of your own. When the decision model points at a written
    function, the REPL runs it. Its parameters are filled as any tool's are, by name from the
    request's answers, then its inputs, and the request's inputs are variables in the REPL as
    well, as a ``dspy.RLM``'s are: copies, so a function changes the world through the tools and
    not by changing a variable.

    Writing only defines: the source is one ``def`` and nothing beside it, under a name that no
    tool has. A function that fails when it is run is taken off the map, and the error is what
    the next request shows as its observation, so that it can be written again.

    Args:
        tools: Functions or ``dspy.Tool`` instances that exist inside the REPL for written code
            to call.
        interpreter_factory: A zero-argument callable that makes the interpreter, as for
            ``dspy.RLM``. None uses ``dspy.settings.interpreter_factory``, then
            ``dspy.PythonInterpreter``.
        name: The environment's name, where there are several.
        description: What the environment is for, where a map offers it by name.
        written: Functions written before, as another ``Repl``'s ``written`` holds them: each
            name with its ``description`` and ``code``. They are defined when the REPL starts.

    Examples:
        ```python
        import dspy
        from dspy.experimental import RDM, TypeSafe

        class Write(dspy.Signature):
            \"\"\"Write one Python function that does what the request still needs, and define it with `define`.\"\"\"

            task: str = dspy.InputField()
            house: dict = dspy.InputField()
            wrote: str = dspy.OutputField(desc="What `define` returned.")

        repl = RDM.Repl(tools=[do, read])                    # what written code may call when it is run
        writer = dspy.RLM(Write, tools=[read, repl.define])  # what the writer may call while it writes
        writer.set_lm(dspy.LM("openai/gpt-5-mini"))

        rdm = RDM(Operate, tools=[*steps, repl, dspy.Tool(writer, name="write", desc="Have a new step written.")], always=["write"])
        rdm.set_lm(TypeSafe("jev-latest"))                   # the decision model points
        rdm(task="Make it three degrees cooler.", house=house)
        print(list(repl.written))                            # ['cool_by_three_degrees']
        ```
    """

    def __init__(
        self,
        tools: list[Callable | Tool] | None = None,
        interpreter_factory: Callable[[], Any] | None = None,
        name: str = "repl",
        description: str = "",
        written: dict[str, dict] | None = None,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.written = copy.deepcopy(dict(written or {}))  # every function written, by name, with its description and its source
        self.kept = Kept(self.written, loaded=self._restart)  # what makes `written` part of the program: saved, loaded, copied
        self.environments = {}  # every environment opened from this one, by name
        self._tools = {tool.name: tool for tool in (tool if isinstance(tool, Tool) else Tool(tool) for tool in tools or [])}
        self._factory = interpreter_factory
        self._interpreter = None
        self._thread = None
        self._lock = threading.Lock()
        self._offered = {}  # the tool made for each function on offer, by its name and source

    @property
    def offers(self) -> list[Tool]:
        """What goes on a map: a tool for every function written that has not failed."""
        offered = {}
        for name, entry in self.written.items():
            if "failed" not in entry:
                key = (name, entry["code"], entry["description"])
                offered[key] = self._offered.get(key) or self._runs(name)
        self._offered = offered
        return list(offered.values())

    def define(self, name: str, description: str, code: str) -> str:
        """Define one function in the REPL and put its name on the list of tools. Nothing is run.

        Args:
            name: The function's name.
            description: One sentence for a menu: what choosing the function does.
            code: The function's source: one ``def`` of that name and nothing beside it. A
                parameter is given the request's answer or input of the same name.

        Returns:
            What was done, in a sentence.

        Raises:
            CodeExecutionError: If the source is not one function of that name, the name is a
                tool's, or the function cannot be defined. Nothing is changed.
        """
        name, description = str(name).strip(), str(description).strip()
        if not name.isidentifier():
            raise CodeExecutionError(f"`{name}` cannot be the name of a function.")
        try:
            code = _strip_code_fences(str(code))  # the fence a language model puts round code
            body = ast.parse(code).body
        except SyntaxError as error:
            raise CodeExecutionError(f"What was written is not Python. {error}") from error
        function = body[0] if len(body) == 1 and isinstance(body[0], ast.FunctionDef) and body[0].name == name else None
        defaults = [] if function is None else [*function.args.defaults, *(default for default in function.args.kw_defaults if default is not None)]
        if function is None or function.decorator_list or not all(isinstance(default, ast.Constant) for default in defaults):
            # Writing only defines: no statement, decorator or default that would run when it is written.
            raise CodeExecutionError(
                f"The source must be one `def {name}(...)` and nothing beside it, with no decorator, and plain values as "
                "defaults: writing a function defines it and runs nothing."
            )
        # The names on the map of the run this is written in, where there is one.
        active = settings.get(ACTIVE)
        taken = active.module._tools(active.run) if active is not None else {}
        if name in self._tools or (name in taken and name not in self.written):
            raise CodeExecutionError(f"`{name}` is already the name of a tool. Give the function another name.")
        try:
            self.execute(code)
        except CodeExecutionError as error:
            if name in self.written:
                self.execute(self.written[name]["code"])  # the function this name had is the one it has still
            raise CodeExecutionError(f"What was written failed. {error}") from error
        self.written[name] = {"description": description, "code": code}
        changed({"by": self.name, "did": "define", "name": name})
        # Said plainly, because a description of what a function does reads as a report that it was done.
        return f"`{name}` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: {description}"

    def remove(self, name: str) -> str:
        """Take a written function off the list of tools, and forget its source.

        Args:
            name: The function's name.
        """
        if self.written.pop(name, None) is None:
            return f"There is no function `{name}`: the functions are {list(self.written)}."
        changed({"by": self.name, "did": "remove", "name": name})
        return f"`{name}` is no longer on the list."

    def show(self) -> str:
        """Every function written so far, with its description and its source, and the other names on the list.

        For whoever writes: what there is already, so that nothing is written twice, and what
        failed, so that it can be written again.
        """
        lines = []
        for name, entry in self.written.items():
            failed = f"\nIt failed and is off the list: {entry['failed']}" if "failed" in entry else ""
            lines.append(f"{name}: {entry['description']}{failed}\n{entry['code']}")
        active = settings.get(ACTIVE)
        others = [name for name in (active.module._tools(active.run) if active is not None else {}) if name not in self.written]
        if others:
            lines.append("Other names on the list, which a function cannot take: " + ", ".join(others))
        return "\n\n".join(lines) or "No function has been written."

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> str:
        """Run code in the REPL and return what it printed.

        An interpreter belongs to the thread that first used it, and a module's tools run on
        many. So the REPL has one thread of its own, and everything is run there, one at a time.

        Raises:
            CodeExecutionError: If the code fails, or if it is asked for by code this REPL is
                running, through a tool: a REPL runs one thing at a time.
        """
        output = self._on_thread(lambda: self._session().execute(code, variables=variables or {}))
        return "" if output is None else str(output).strip()

    def spawn(self, name: str, description: str = "", tools: list[Callable | Tool] | None = None) -> "Repl":
        """A new environment: another REPL from the same factory, isolated from this one.

        A function written in one environment does not exist in another. To route to it, put a
        module that runs in it on a map under its name:
        ``dspy.Tool(RDM(signature, tools=[environment, ...]), name=environment.name, desc=environment.description)``.

        Args:
            name: The environment's name.
            description: What the environment is for.
            tools: The functions that exist inside it. None gives it this environment's.
        """
        child = Repl(list(self._tools.values()) if tools is None else tools, self._factory, name=name, description=description)
        self.environments[name] = child
        return child

    def shutdown(self) -> None:
        """Shut down the interpreter, and those of the environments opened from this one."""
        for child in self.environments.values():
            child.shutdown()
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is not None:
            if self._interpreter is not None:
                thread.submit(self._interpreter.shutdown).result()
                self._interpreter = None
            thread.shutdown()

    def __enter__(self) -> "Repl":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __getstate__(self):
        # What an interpreter, a thread and a lock are cannot be pickled, and need not be: a REPL
        # that is read back starts when it is first used, with the functions that were written.
        state = dict(self.__dict__)
        state.update(_interpreter=None, _thread=None, _lock=None, _offered={})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self.kept = Kept(self.written, loaded=self._restart)

    def __deepcopy__(self, memo) -> "Repl":
        # A copy is an environment of its own: the same tools and sources, and an interpreter
        # that is made, and has the sources defined in it, when the copy is first used.
        copied = Repl(list(self._tools.values()), self._factory, name=self.name, description=self.description, written=self.written)
        memo[id(self)] = copied
        copied.environments = {name: copy.deepcopy(child, memo) for name, child in self.environments.items()}
        return copied

    def _restart(self):
        """Start again from what is written: after a load, the REPL holds the functions that were loaded and no others."""
        environments, self.environments = self.environments, {}
        try:
            self.shutdown()
        finally:
            self.environments = environments

    def _runs(self, name):
        """The tool that runs a written function in the REPL.

        It takes the parameters the function takes, so the module fills them as it fills any
        tool's: by name from the request's answers, then its inputs. One that names neither is
        an observation, since a language model named it and can write the function again.
        """
        from dspy.predict.rdm import Call

        takes = _takes(self.written[name]["code"], name)

        def run(call, **arguments):
            missing = [parameter for parameter, required in takes.items() if required and parameter not in arguments]
            if missing:
                given = sorted({*call.inputs, *(call.prediction.keys() if call.prediction is not None else ())})
                return f"`{name}` was not run: it takes {missing}, which name no answer or input of the request. A parameter is filled by name from {given}."
            variables = _variables(call.inputs)
            try:
                if not arguments:
                    return self.execute(f"print({name}())", variables=variables)
                return self.execute(f"print({name}(**{_ARGUMENTS}))", variables={**variables, _ARGUMENTS: arguments})
            except CodeExecutionError as error:
                # A function that fails would be pointed at again: it comes off the map, and the error is
                # what is observed. Its source is kept, marked, for whoever writes it again.
                if name in self.written:
                    self.written[name]["failed"] = str(error)
                    changed({"by": self.name, "did": "fail", "name": name})
                return f"`{name}` failed and was taken off the list. {error}"

        # Every parameter may be left out as far as the module is concerned: what is missing is said above.
        run.__signature__ = inspect.Signature([
            inspect.Parameter("call", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Call),
            *(inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY, default=None) for parameter in takes),
        ])
        return Tool(run, name=name, desc=self.written[name]["description"])

    def _session(self):
        """The interpreter, made on first use with the tools and whatever was written before. Called on the REPL's own thread."""
        if self._interpreter is None:
            interpreter = _create_interpreter(self._factory)
            interpreter.tools.update({name: RLM._make_interpreter_tool(tool) for name, tool in self._tools.items()})
            for entry in self.written.values():
                interpreter.execute(entry["code"])
            self._interpreter = interpreter
        return self._interpreter

    def _on_thread(self, call):
        """Run a call on the REPL's own thread, with the caller's DSPy settings."""
        if id(self) in _RUNNING.get():
            # Code this REPL is running led here, and the REPL's one thread is busy with that code.
            raise CodeExecutionError(
                f"The REPL `{self.name}` is running the code that asked for this, and runs one thing at a time: "
                "a function written here cannot be run from inside one that is running."
            )
        with self._lock:
            if self._thread is None:
                self._thread = ThreadPoolExecutor(max_workers=1)
            thread = self._thread

        def running():
            _RUNNING.set(_RUNNING.get() | {id(self)})
            return call()

        return thread.submit(contextvars.copy_context().run, running).result()


def _takes(code, name):
    """The parameters of the function a source defines under a name, each with whether it must be given."""
    try:
        defined = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == name]
    except SyntaxError:
        return {}
    if not defined:
        return {}
    arguments = defined[-1].args
    positional = [*arguments.posonlyargs, *arguments.args]
    takes = {argument.arg: position < len(positional) - len(arguments.defaults) for position, argument in enumerate(positional)}
    takes.update({argument.arg: default is None for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)})
    return takes


def _variables(inputs):
    """The inputs of a request that can be given to a REPL as variables: those that are plain data."""
    variables = {}
    for name, value in inputs.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        variables[name] = value
    return variables
