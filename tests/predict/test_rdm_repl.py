"""Tests for RDM.Repl: a REPL where what a language model writes can be pointed at by a decision model.

Test organization:
- One REPL: what is written stays, and can be pointed at
- A Repl in a program: saved, loaded, copied, and used from any thread
- Several REPLs: environments that are isolated, and routed to by name

Every test runs on each interpreter there is: the in-process one of dspy-interpreters;
`dspy.PythonInterpreter`, the Deno sandbox (`--deno`); and Monty, where it is installed.
"""

import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest

import dspy
from dspy.experimental import RDM
from dspy.primitives.code_interpreter import CodeExecutionError
from dspy.utils.dummies import DummyLM


def interpreters():
    """Every interpreter the helper is tested on, each as a factory."""
    params = [pytest.param(dspy.PythonInterpreter, id="deno", marks=pytest.mark.deno)]
    try:
        from dspy_interpreters import InProcessInterpreter

        params.insert(0, pytest.param(InProcessInterpreter, id="in-process"))
    except ImportError:  # dspy-interpreters is a development dependency
        pass
    try:
        from dspy_interpreters.monty import MontyInterpreter

        params.append(pytest.param(MontyInterpreter, id="monty"))
    except ImportError:  # Monty is an extra of dspy-interpreters
        params.append(pytest.param(None, id="monty", marks=pytest.mark.skip(reason="Monty is not installed")))
    return params


INTERPRETERS = interpreters()


class Client:
    """A stand-in System One client that points at the names of a script, in order."""

    supports_decision_requests = True
    cache = True
    model = "jev-test"

    def __init__(self, *script, **chosen):
        self.script, self.chosen, self.calls, self.history, self.callbacks = list(script), chosen, [], [], []

    def __call__(self, state, questions):
        self.calls.append({"inputs": state["inputs"], "offered": list(questions["next"]["criteria"])})
        pointed = self.script.pop(0)
        assert pointed in questions["next"]["criteria"], f"{pointed!r} is not on the map: {list(questions['next']['criteria'])}"
        # Any other question of the request is answered with the option this client was told to choose.
        chosen = {"next": pointed, **{name: str(option) for name, option in self.chosen.items()}}
        return {name: {"probabilities": {label: float(label == chosen[name]) for label in question["criteria"]}, "confidence": 1.0}
                for name, question in questions.items()}


class Operate(dspy.Signature):
    """Operate a heater, one step at a time."""

    request: str = dspy.InputField()
    world: dict = dspy.InputField()
    observation: str = dspy.InputField()
    next: dspy.Tool = dspy.OutputField(desc="What should happen next for `inputs.request`?")


class Write(dspy.Signature):
    """Write one Python function that does what the request still needs."""

    request: str = dspy.InputField()
    world: dict = dspy.InputField()
    observation: str = dspy.InputField(desc="How the last step went. If a function failed, its error is here.")
    name: str = dspy.OutputField(desc="The function's name.")
    description: str = dspy.OutputField(desc="One sentence for a menu: what choosing it does.")
    code: str = dspy.OutputField(desc="The function's source.")


class World:
    """A heater, and the two tools that exist for it inside a REPL."""

    def __init__(self):
        self.state = {"heat": 21}

    def cooler(self) -> str:
        """Make it one degree cooler, once."""
        self.state["heat"] -= 1
        return f"heat is {self.state['heat']}"

    def read(self) -> str:
        """The state, as JSON text."""
        return json.dumps(self.state)


COOL_THREE = "def cool_three():\n    for _ in range(3):\n        cooler()\n    return 'cooled by three: ' + read()"
WROTE = {"name": "cool_three", "description": "Make it three degrees cooler.", "code": COOL_THREE}
WRITTEN = "`cool_three` was written and is now on the list. It has not been run, so nothing has changed yet. It does this: Make it three degrees cooler."


def environment(world, interpreter, **kwargs):
    return RDM.Repl(tools=[world.cooler, world.read], interpreter_factory=interpreter, **kwargs)


def writer(repl, *answers, name="write"):
    """A tool that writes: a language model gives a function in one call, and the tool defines it in the REPL."""
    gives = dspy.Predict(Write)
    gives.set_lm(DummyLM(list(answers)))

    def write(request, world, observation):
        wrote = gives(request=request, world=world, observation=observation)
        try:
            return repl.define(wrote.name, wrote.description, wrote.code)
        except CodeExecutionError as error:
            return f"Nothing was put on the list. {error}"

    write.gives = gives
    return dspy.Tool(write, name=name, desc="Have a new tool written in code: nothing listed does what is still needed.")


def house(repl, *answers):
    write = writer(repl, *answers)
    return RDM(Operate, tools=[write, repl]), write


# One REPL: what is written stays, and can be pointed at


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_what_is_written_is_defined_in_the_repl_and_is_on_the_map(interpreter):
    world = World()
    with environment(world, interpreter) as repl:
        rdm, write = house(repl, WROTE)
        client = Client("write", "cool_three", "finish", "cool_three", "finish")
        rdm.set_lm(client)
        result = rdm(request="Three degrees cooler.", world=world.state)
        # The decision model pointed at `write`, a tool like any other, which called `repl.define`.
        assert "Three degrees cooler." in json.dumps(write.func.gives.lm.history[0]["messages"]) and len(write.func.gives.lm.history) == 1
        # What was defined is on the map from the next request: the map holds what the REPL offers then.
        assert [call["offered"] for call in client.calls[:2]] == [["write", "finish"], ["write", "cool_three", "finish"]]
        assert client.calls[1]["inputs"]["observation"] == WRITTEN
        # The decision model pointed at the name, and the REPL ran it, with the tools: the loop a decision model cannot make.
        assert world.state == {"heat": 18} and result.observations["next"][1] == ["cool_three", 'cooled by three: {"heat": 18}']
        assert repl.written == {"cool_three": {"description": "Make it three degrees cooler.", "code": COOL_THREE}}
        # The change is recorded in the trajectory, on the request whose tool made it.
        assert [entry.get("changes") for entry in result.trajectory] == [[{"by": "repl", "did": "define", "name": "cool_three"}], None, None]
        # `show` is for whoever writes next: what is written already, with its source, and the names it cannot take.
        assert repl.show() == f"cool_three: Make it three degrees cooler.\n{COOL_THREE}"
        # It is kept: the next run is offered it from its first request, and nothing is written again.
        rdm(request="And three more.", world=world.state)
        assert client.calls[3]["offered"] == ["write", "cool_three", "finish"] and world.state == {"heat": 15}
        assert len(write.func.gives.lm.history) == 1
        # `remove` takes a function off the map and forgets it.
        assert repl.remove("cool_three") == "`cool_three` is no longer on the list." and repl.written == {}
        assert repl.remove("cool_three").startswith("There is no function `cool_three`") and [tool.name for tool in repl.offers] == []


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_function_that_fails_is_taken_off_the_map_and_can_be_written_again(interpreter):
    world = World()
    broken = {**WROTE, "code": "def cool_three():\n    return cooler() + str(1 // 0)"}
    with environment(world, interpreter) as repl:
        rdm, write = house(repl, broken, WROTE)
        client = Client("write", "cool_three", "write", "cool_three", "finish")
        rdm.set_lm(client)
        rdm(request="Three degrees cooler.", world=world.state)
        observed = [call["inputs"]["observation"] for call in client.calls]
        # A function that fails when it is run comes off the map, so it is not pointed at again.
        assert "cool_three" in client.calls[1]["offered"] and "cool_three" not in client.calls[2]["offered"]
        assert observed[2].startswith("`cool_three` failed and was taken off the list.") and "ZeroDivisionError" in observed[2]
        # The writer is given the error, as the request's observation, and what it writes next takes the name.
        assert "ZeroDivisionError" in json.dumps(write.func.gives.lm.history[1]["messages"])
        assert world.state == {"heat": 17} and repl.written["cool_three"] == {"description": "Make it three degrees cooler.", "code": COOL_THREE}
    # Its source is kept, marked as failed, for whoever writes it again.
    with environment(world, interpreter) as repl:
        rdm, write = house(repl, broken)
        rdm.set_lm(Client("write", "cool_three", "finish"))
        result = rdm(request="Three degrees cooler.", world=world.state)
        assert "ZeroDivisionError" in repl.written["cool_three"]["failed"] and repl.offers == []
        assert result.trajectory[1]["changes"] == [{"by": "repl", "did": "fail", "name": "cool_three"}]
        assert "It failed and is off the list:" in repl.show() and "ZeroDivisionError" in repl.show()


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_writing_only_defines_and_what_is_refused_changes_nothing(interpreter):
    world = World()
    with environment(world, interpreter) as repl:
        assert repl.define(**WROTE) == WRITTEN
        refused = {
            # Source that would run something when it is written: a call beside the function, a decorator, a default that calls.
            COOL_THREE + "\ncool_three()": "The source must be one `def cool_three(...)`",
            "@cooler\n" + COOL_THREE: "The source must be one `def cool_three(...)`",
            "def cool_three(now=cooler()):\n    return now": "The source must be one `def cool_three(...)`",
            "x = 1": "The source must be one `def cool_three(...)`",
            "def cool_three(:\n    return": "What was written is not Python.",
        }
        for code, reason in refused.items():
            with pytest.raises(CodeExecutionError, match=reason.replace("(", "\\(").replace(")", "\\)")):
                repl.define("cool_three", "Something else.", code)
        # A name the REPL's own tools have: the function would take the tool's place for everything written before.
        with pytest.raises(CodeExecutionError, match="`cooler` is already the name of a tool"):
            repl.define("cooler", "Cool twice.", "def cooler():\n    return 'not the tool'")
        with pytest.raises(CodeExecutionError, match="cannot be the name of a function"):
            repl.define("cool three", "Cool.", COOL_THREE)
        # Nothing ran while it was written, and the function written first is still the one the name has.
        assert repl.written == {"cool_three": {"description": "Make it three degrees cooler.", "code": COOL_THREE}}
        assert world.state == {"heat": 21} and repl.execute("print(cool_three())") == 'cooled by three: {"heat": 18}'
        # A fence round the source, as a language model writes code, is taken off.
        assert repl.define("cool_three", "Make it three degrees cooler.", f"```python\n{COOL_THREE}\n```") == WRITTEN and repl.written["cool_three"]["code"] == COOL_THREE
        # In a run, a name that a tool on the map has is refused too, and the writer is told.
        rdm, _ = house(repl, {**WROTE, "name": "write", "code": "def write():\n    return 1"})
        client = Client("write", "finish")
        rdm.set_lm(client)
        rdm(request="Write.", world=world.state)
        assert client.calls[1]["inputs"]["observation"] == "Nothing was put on the list. `write` is already the name of a tool. Give the function another name."


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_written_function_sees_the_inputs_of_the_request_that_pointed_at_it(interpreter):
    world = World()
    with environment(world, interpreter) as repl:
        repl.define("echo", "Say what was asked.", "def echo():\n    return 'asked: ' + request")
        rdm = RDM(Operate, tools=[repl])
        rdm.set_lm(Client("echo", "finish", "echo", "finish"))
        first = rdm(request="one thing", world=world.state)
        # The request's inputs are variables in the REPL each time a function is run: not those of the run it was written in.
        second = rdm(request="another thing", world=world.state)
        assert first.observations["next"][0] == ["echo", "asked: one thing"]
        assert second.observations["next"][0] == ["echo", "asked: another thing"]


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_written_function_takes_its_parameters_as_any_tool_does(interpreter):
    class OperateBy(Operate):
        """Operate a heater, one step at a time."""

        degrees: Literal[1, 2, 3] = dspy.OutputField(desc="By how many degrees does `inputs.request` ask?")

    world = World()
    with environment(world, interpreter) as repl:
        repl.define("cool_by", "Make it cooler by the degrees asked for.", "def cool_by(degrees, request, times=1):\n    for _ in range(degrees * times):\n        cooler()\n    return request + ' ' + read()")
        rdm = RDM(OperateBy, tools=[repl])
        rdm.set_lm(Client("cool_by", "finish", degrees=3))
        result = rdm(request="Three cooler.", world=world.state)
        # `degrees` is an answer of the request that pointed at the function, and `request` an input of it: each
        # is given by name, as to any tool. `times` names neither and has a default, so it is left alone.
        assert world.state == {"heat": 18} and result.observations["next"][0] == ["cool_by", 'Three cooler. {"heat": 18}']
        # So one function serves every number: what it is told is a question, answered by the decision model.
        rdm.set_lm(Client("cool_by", "finish", degrees=2))
        rdm(request="Two cooler.", world=world.state)
        assert world.state == {"heat": 16} and list(repl.written) == ["cool_by"]
        # A parameter that names no answer or input is said, as an observation: a language model named it and can write it again.
        repl.define("cool_some", "Make it cooler.", "def cool_some(amount):\n    return amount")
        rdm.set_lm(Client("cool_some", "finish", degrees=1))
        said = rdm(request="Cooler.", world=world.state).observations["next"][0][1]
        assert said.startswith("`cool_some` was not run: it takes ['amount'], which name no answer or input of the request.")


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_dspy_rlm_writes_by_calling_define_as_its_tool(interpreter):
    world = World()
    with environment(world, interpreter) as repl:
        # An RLM writes in a sandbox of its own. It is given what is safe to try, `read`, and `define`, which only
        # defines: a writer that could call `cooler` would act while it writes.
        code = f"SUBMIT(wrote=define(name='cool_three', description='Make it three degrees cooler.', code={COOL_THREE!r}))"
        rlm = dspy.RLM("request, world -> wrote", tools=[world.read, repl.define], interpreter_factory=interpreter, max_iters=2)
        rlm.set_lm(DummyLM([{"reasoning": "Write it and define it.", "code": f"```python\n{code}\n```"}]))
        rdm = RDM(Operate, tools=[dspy.Tool(rlm, name="write", desc="Have a new tool written in code."), repl])
        client = Client("write", "cool_three", "finish")
        rdm.set_lm(client)
        rdm(request="Three degrees cooler.", world=world.state)
        # What the RLM submitted is the observation, and the function is on the map and runs in the shared REPL.
        assert WRITTEN in client.calls[1]["inputs"]["observation"] and client.calls[1]["offered"] == ["write", "cool_three", "finish"]
        assert world.state == {"heat": 18} and repl.written["cool_three"]["code"] == COOL_THREE


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_two_repls_share_one_map(interpreter):
    world = World()
    note = {"name": "note", "description": "Note what was asked.", "code": "def note():\n    return 'noted: ' + request"}
    with environment(world, interpreter) as heating, environment(world, interpreter, name="notes") as notes:
        # Each has a tool that writes into it, under a name of its own. What both offer is on the one map.
        rdm = RDM(Operate, tools=[writer(heating, WROTE, name="write_heating"), writer(notes, note, name="write_note"), heating, notes])
        client = Client("write_heating", "write_note", "cool_three", "note", "finish")
        rdm.set_lm(client)
        result = rdm(request="Cooler, and note it.", world=world.state)
        assert client.calls[-1]["offered"] == ["write_heating", "write_note", "cool_three", "note", "finish"]
        assert (list(heating.written), list(notes.written)) == (["cool_three"], ["note"])
        assert world.state == {"heat": 18} and result.observations["next"][3] == ["note", "noted: Cooler, and note it."]
        # A name that two of them offer is a mistake of the program, and is said.
        notes.define(**WROTE)
        with pytest.raises(ValueError, match="\\['cool_three'\\] are offered by `Repl` and are already names on the map"):
            rdm(request="Again.", world=world.state)


# A Repl in a program: saved, loaded, copied, and used from any thread


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_what_was_written_is_saved_and_loaded_with_the_program(interpreter, tmp_path):
    world = World()

    def program():
        repl = environment(world, interpreter)
        return RDM(Operate, tools=[repl]), repl

    rdm, repl = program()
    fresh, restored = program()
    try:
        repl.define(**WROTE)
        rdm.save(tmp_path / "program.json")
        # A program built the same way and loaded has the functions, defined in its own REPL, with nothing written.
        restored.define("stale", "Something written before the load.", "def stale():\n    return 'stale'")
        fresh.load(tmp_path / "program.json")
        assert restored.written == repl.written and restored is not repl
        client = Client("cool_three", "finish")
        fresh.set_lm(client)
        fresh(request="Three degrees cooler.", world=world.state)
        assert client.calls[0]["offered"] == ["cool_three", "finish"] and world.state == {"heat": 18}
        with pytest.raises(CodeExecutionError, match="NameError"):
            restored.execute("print(stale())")
        # The source is plain data too: another REPL can be given it.
        with environment(world, interpreter, written=json.loads(json.dumps(repl.written))) as given:
            assert given.execute("print(cool_three())") == 'cooled by three: {"heat": 15}'
    finally:
        repl.shutdown()
        restored.shutdown()


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_whole_program_with_a_repl_is_saved_and_read_back(interpreter, tmp_path):
    pytest.importorskip("cloudpickle")
    world = World()
    with environment(world, interpreter) as repl:
        rdm = RDM(Operate, tools=[repl])
        repl.define("first", "Say first.", "def first():\n    return 'first'")
        repl.execute("print(first())")  # an interpreter, a thread and a lock are in use, and none of them is saved
        rdm.save(tmp_path / "program", save_program=True)
    loaded = dspy.load(tmp_path / "program", allow_pickle=True)
    restored = loaded.sources[0]
    try:
        # The program read back has a REPL that starts when it is first used, with the functions that were written.
        assert restored.written == {"first": {"description": "Say first.", "code": "def first():\n    return 'first'"}}
        loaded.set_lm(Client("first", "finish"))
        assert loaded(request="Say it.", world={}).observations["next"][0] == ["first", "first"]
    finally:
        restored.shutdown()


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_copy_of_the_program_has_a_repl_of_its_own(interpreter):
    world = World()
    with environment(world, interpreter) as repl:
        rdm, _ = house(repl, WROTE)
        repl.define("first", "Say first.", "def first():\n    return 'first'")
        # A copy, as an optimizer makes, starts with the same functions and writes into its own REPL.
        twin = rdm.deepcopy()
        copied = twin.sources[0]
        try:
            assert copied is not repl and copied.written == repl.written
            copied.define(**WROTE)
            client = Client("cool_three", "finish")
            twin.set_lm(client)
            twin(request="Three degrees cooler.", world=world.state)
            assert client.calls[0]["offered"] == ["write", "first", "cool_three", "finish"] and world.state == {"heat": 18}
            assert list(repl.written) == ["first"] and [tool.name for tool in repl.offers] == ["first"]
            # Nothing else an environment holds is shared either.
            repl.execute("kept = 7")
            with pytest.raises(CodeExecutionError, match="NameError"):
                copied.execute("print(kept)")
        finally:
            copied.shutdown()


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_a_repl_is_run_on_one_thread_whatever_thread_asks(interpreter):
    world = World()
    with environment(world, interpreter) as heat, environment(world, interpreter, name="heating") as heating:
        heat.execute("def double(n):\n    return n * 2")
        # An interpreter belongs to one thread. A module's tools and batches run on many: all of them reach the same REPL.
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(lambda n: heat.execute(f"print(double({n}))"), range(4))) == ["0", "2", "4", "6"]
        thread = threading.Thread(target=lambda: heat.execute("kept = 7"))
        thread.start()
        thread.join()
        assert heat.execute("print(kept)") == "7"
        # Each REPL has a thread of its own, whatever the REPLs are called.
        with pytest.raises(CodeExecutionError, match="NameError"):
            heating.execute("print(kept)")


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_code_a_repl_is_running_cannot_call_back_into_it_and_can_reach_another_environment(interpreter):
    said = []

    def again() -> str:
        """Ask the REPL for something from the tool it called."""
        try:
            return repl.execute("print(inner())")
        except CodeExecutionError as error:
            said.append(str(error))
            return "refused"

    def other() -> str:
        """Run a function written in another environment."""
        return room.execute("print(inner())")

    def elsewhere() -> str:
        """Ask the REPL for something from another thread, as a tool that makes sub-calls at once does."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                return pool.submit(contextvars.copy_context().run, lambda: repl.execute("print(1)")).result(timeout=20)
            except CodeExecutionError as error:
                said.append(str(error))
                return "refused"

    with RDM.Repl(tools=[elsewhere, again, other], interpreter_factory=interpreter) as repl:
        room = repl.spawn("room")
        for environment in (repl, room):
            environment.define("inner", "Say so.", f"def inner():\n    return 'inner ran in {environment.name}'")
        # A REPL runs one thing at a time. Code it is running cannot call back into it through a tool, on the
        # tool's thread or on another, where it could never answer: this is said, the same on every interpreter.
        assert repl.execute("print(again())") == "refused" and repl.execute("print(elsewhere())") == "refused"
        assert len(said) == 2 and all("runs one thing at a time" in error for error in said)
        # One written function calls another directly, and another environment, with its own interpreter, can be reached.
        assert repl.execute("def outer():\n    return 'outer, then ' + inner()\nprint(outer())") == "outer, then inner ran in repl"
        assert repl.execute("print(other())") == "inner ran in room"


# Several REPLs: environments that are isolated, and routed to by name


@pytest.mark.parametrize("interpreter", INTERPRETERS)
def test_environments_are_isolated_and_a_name_on_the_map_routes_to_one(interpreter):
    world = World()
    lights_off = {"name": "lights_off", "description": "Turn the lights off.", "code": "def lights_off():\n    return 'lights off'"}
    with environment(world, interpreter) as repl:
        heating = repl.spawn("heating", "Anything about the temperature.")
        lights = repl.spawn("lights", "Anything about the lights.")
        client = Client("heating", "write", "cool_three", "finish", "lights", "write", "lights_off", "finish", "finish")
        with dspy.context(lm=client):
            # Each environment is a module over its own map. On the root's map a name stands for a whole environment.
            rooms = [
                dspy.Tool(RDM(Operate, tools=[writer(room, wrote), room]), name=room.name, desc=room.description)
                for room, wrote in ((heating, WROTE), (lights, lights_off))
            ]
            root = RDM(Operate, tools=rooms)
            result = root(request="Three degrees cooler, and the lights off.", world=world.state)
        # The root was offered the environments, and pointing at one put the request to it, one level down.
        assert client.calls[0]["offered"] == ["heating", "lights", "finish"] and world.state == {"heat": 18}
        assert [(entry["depth"], entry["outputs"]["next"]["value"]) for entry in result.trajectory] == [
            (0, "heating"), (1, "write"), (1, "cool_three"), (1, "finish"),
            (0, "lights"), (1, "write"), (1, "lights_off"), (1, "finish"), (0, "finish"),
        ]
        # Each has its own interpreter, and what was written in one is there and nowhere else: not in
        # another's REPL, and not on another's map.
        assert list(heating.written) == ["cool_three"] and list(lights.written) == ["lights_off"] and repl.written == {}
        assert heating.execute("print(cool_three())") == 'cooled by three: {"heat": 15}'
        assert lights.execute("print(lights_off())") == "lights off"
        for room, other in ((lights, "cool_three"), (heating, "lights_off"), (repl, "cool_three")):
            with pytest.raises(CodeExecutionError, match="NameError"):
                room.execute(f"print({other}())")
        assert client.calls[5]["offered"] == ["write", "finish"]
        # So is anything else an environment holds.
        heating.execute("kept = 7")
        assert heating.execute("print(kept)") == "7"
        with pytest.raises(CodeExecutionError, match="NameError"):
            lights.execute("print(kept)")
        # They are given the same tools, which act on one world.
        assert lights.execute("print(cooler())") == "heat is 14" and repl.environments == {"heating": heating, "lights": lights}
