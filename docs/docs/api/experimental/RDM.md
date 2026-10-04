# dspy.experimental.RDM

!!! warning "Experimental API"
    `RDM` is experimental and may change without warning.

`RDM` (Recursive Decision Model) applies the idea of [`dspy.RLM`](../modules/RLM.md) to decision models: models such as Jev, through [`dspy.experimental.TypeSafe`](DecisionTypes.md), that answer typed questions with a decision and a probability and write nothing. An RDM answers a signature as [`dspy.Predict`](../modules/Predict.md) does, and lets each answer run a tool. A tool can ask the module again about a piece of what it was given, so the decisions recurse through a state that no single request shows whole. With code as the only tools it runs with no generative model at all, and a generative model can be one of the tools.

This implements the approach described in ["Recursive Decision Models" (Weitekamp, 2026)](https://raw.works/recursive-decision-models/), which applies to decision models the idea of ["Recursive Language Models" (Zhang, Kraska, Khattab, 2025)](https://arxiv.org/abs/2512.24601).

## One request, and what its answers run

A request is a signature whose outputs are decisions. An output named after a tool calls it on a yes, and the tool's parameters are filled by name from the other answers of the same request, then from its inputs.

```python
from typing import Literal

import dspy
from dspy.experimental import RDM, Noul, TypeSafe

house = {"tv": "off", "lights": "on", "door": "open", "lock": "unlocked"}

def switch(device, to):
    """Set one device to one state."""
    if device == "lock" and to == "locked" and house["door"] == "open":
        return "lock: refused, the door is open"
    house[device] = to
    return f"{device}: now {to}"

class Act(dspy.Signature):
    """Operate a smart home."""

    task: str = dspy.InputField()
    house: dict = dspy.InputField()
    switch: Noul = dspy.OutputField(desc="Does `inputs.task` ask for one device in `inputs.house` to be set to another state?")
    device: Literal["tv", "lights", "door", "lock"] = dspy.OutputField(desc="Which device does `inputs.task` mean?")
    to: Literal["on", "off", "open", "closed", "locked", "unlocked"] = dspy.OutputField(desc="Which state does `inputs.task` ask for?")

act = RDM(Act, tools=[switch])
act.set_lm(TypeSafe("jev-latest"))  # pip install "dspy[typesafe]" and set TYPESAFE_API_KEY

result = act(task="Turn the TV on.", house=house)
print(result.observations)
# {'switch': 'tv: now on'}
```

That is one request with three questions. The decision model does not write an argument: it chooses one, as another answer of the request, and the code passes it on. With no tools, `RDM(Act)` is `dspy.Predict(Act)`. Every decision output needs a `desc`: it is the question put to the model. Where one request can decide, use one.

## A decision at every step

An output named after a tool is for a request that decides once whether to run it. Some requests cannot be decided at once, because what a step does changes what the next should be: that is what an output typed `dspy.Tool` is for. An output typed `dspy.Tool` points at one of the module's tools, as [`dspy.ReAct`](../modules/ReAct.md) lets a language model choose its next tool. The tools are a map: each has a name and a description for the model, and a function for the code. On this page "the map" is always that: the module's tools, by name, as the pointer is offered them. What the models are shown calls it a list.

```python
from functools import partial

class Operate(dspy.Signature):
    """Operate a smart home, one step at a time."""

    task: str = dspy.InputField()
    house: dict = dspy.InputField()
    observation: str = dspy.InputField(desc="What the last step did.")
    next: dspy.Tool = dspy.OutputField(desc="What should happen next for `inputs.task`?")

steps = [
    dspy.Tool(partial(switch, device, to), name=f"{device} {to}", desc=f"Set the {device} to {to}.")
    for device, to in [("tv", "on"), ("tv", "off"), ("lights", "on"), ("lights", "off"), ("door", "closed"), ("lock", "locked")]
]
rdm = RDM(Operate, tools=steps)
rdm.set_lm(TypeSafe("jev-latest"))

result = rdm(task="Lock the front door.", house=house)
print(house["door"], house["lock"])
# closed locked
```

Each request offers the tools by name and description, with `finish`. The loop is the module's, not the model's: the tool pointed at runs, and the same closed question is asked again about the state as it now is: `house` is shown as it is now, because the tools change the same dict the call was given, and `observation` shows what the tool returned. `result.observations["next"]` holds what was done, most often:

```
[['lock locked', 'lock: refused, the door is open'], ['door closed', 'door: now closed'], ['lock locked', 'lock: now locked'], ['finish', None]]
```

Nothing in the code says to close the door before locking it. About one run in five, Jev closes the door first and is never refused. The run ends when the model points at `finish`, and `max_iters` bounds it. `RDM(..., finish=False)` leaves `finish` out, for a game, say, where stopping is not the model's to choose: the loop then ends at a limit, at a repeat, or at a tool named `finish` if you give it one.

## Keeping what was done

The map can change. A tool put in `rdm.tools` is offered from the next run.

```python
house.update(tv="on", lights="on", door="open", lock="unlocked")
result = rdm(task="Good night: everything off, and lock up.", house=house)
done = [name for name, observed in result.observations["next"] if name != "finish" and "refused" not in observed]
print(house)
# {'tv': 'off', 'lights': 'off', 'door': 'closed', 'lock': 'locked'}

def good_night(call: RDM.Call):
    """Take the steps that worked, in the order they were taken."""
    return [call.tools[name]() for name in done]

rdm.tools["good night"] = dspy.Tool(good_night, name="good night", desc=f"Good night: everything off, and lock up ({', '.join(done)}).")
house.update(tv="on", lights="on", door="open", lock="unlocked")
result = rdm(task="Night!", house=house)
print(house)
# {'tv': 'off', 'lights': 'off', 'door': 'closed', 'lock': 'locked'}
```

Most runs, Jev now points at `good night` and then at `finish`: one point does what took several. In the others it takes the steps one by one, to the same end. Code keeps it here, between runs. A tool that takes a parameter annotated `RDM.Call` is given the request that pointed at it, and `call.tools` is the run's map. Each run points at its own copy of `rdm.tools`, so runs on other threads are not affected: inside a run, a tool puts a tool in `call.tools` to offer it from the run's next request. So the decision model can choose what to keep by pointing at a tool that keeps, and a generative model inside a tool can write the description.

## Recursion: a tool that asks the module again

Calling `call.rdm` inside a tool is a sub-call: one level deeper, in the same run, with its prediction returned to the tool. The text below is nineteen lines, which one request reads whole; it is small so that the example can be read. Recursion is for a state that no request can hold, or where an answer decides what is asked next.

```python
import codecs
import this  # importing it prints the Zen of Python

zen = codecs.decode(this.s, "rot13")

class Search(dspy.Signature):
    """Search a text for the answer to a question."""

    text: str = dspy.InputField()
    question: str = dspy.InputField()
    look: Noul = dspy.OutputField(desc="Is the answer to `inputs.question` stated in `inputs.text`?")

def look(text, question, call: RDM.Call):
    """On a yes, ask the same question about each half, one level down."""
    lines = text.splitlines()
    if len(lines) == 1:
        return lines
    halves = lines[: len(lines) // 2], lines[len(lines) // 2 :]
    return [line for half in halves for line in call.rdm(text="\n".join(half), question=question).observations.get("look", [])]

search = RDM(Search, tools=[look], max_depth=6)
search.set_lm(TypeSafe("jev-latest"))
print(search(text=zen, question="How many obvious ways should there be to do something?").observations["look"])
# ['There should be one-- and preferably only one --obvious way to do it.']
```

Each request is a yes or a no about at most the text it was given. `call.rdm` is the module that is running, so a copy an optimizer makes recurses into itself.

`call.rdm.batch` makes several sub-calls at once, each a dict of keyword arguments: `signature=` puts other questions, and `max_depth=0` asks for the answers only, with no tool run. What no request should show is given to the run as `state=` and read as `call.state`:

```python
class Ask(dspy.Signature):
    """Decide whether to look for the answer to a question."""

    question: str = dspy.InputField()
    each: Noul = dspy.OutputField(desc="Could one line of a text state the answer to `inputs.question`?")

def each(question, call: RDM.Call):
    """Put the question to every line of the text at once, for the answers only."""
    lines = [line for line in call.state["text"].splitlines() if line]
    verdicts = call.rdm.batch([{"signature": Search, "text": line, "question": question, "max_depth": 0} for line in lines])
    said_yes = RDM.Filter()(zip(lines, (verdict.look for verdict in verdicts)))  # the yeses, likeliest first
    return [line for line, probability in said_yes]

lines = RDM(Ask, tools=[each])
lines.set_lm(TypeSafe("jev-latest"))
print(lines(question="How many obvious ways should there be to do something?", state={"text": zen}).observations["each"])
# ['There should be one-- and preferably only one --obvious way to do it.']
```

No request shows the whole text: the first shows the question alone, and each sub-call one line.

## Choosing the next question

A sub-call can put other questions: `signature=` replaces the module's own for that call. Here each answer decides which options the next question has.

```python
from dspy.experimental import Choice

taxonomy = {
    "Products": ["Animals & Pet Supplies", "Furniture", "Kitchen"],
    "Animals & Pet Supplies": ["Cat Supplies", "Dog Supplies"],
    "Cat Supplies": ["Cat Beds", "Cat Toys", "Cat Window Perches"],
}

def question(category):
    """Which category under this one fits the listing?"""
    options = Choice[tuple((child, "") for child in taxonomy[category])]
    return dspy.Signature("listing -> category").with_updated_fields(
        "category", type_=options, desc="Which category best matches the product in `inputs.listing`?"
    )

def category(category, listing, call: RDM.Call):
    """Put the question under the category chosen, one level down. Comes to the leaf it ends at."""
    if category not in taxonomy:
        return category
    return call.rdm(signature=question(category), listing=listing).observations["category"]

classify = RDM(question("Products"), tools=[category])
classify.set_lm(TypeSafe("jev-latest"))
print(classify(listing="A padded shelf that suction-cups to a window, so a cat can sun itself.").observations["category"])
# Cat Window Perches
```

The call's own answer, `.category`, is its own choice: the first. What its tool came to is in `observations`. This walk is greedy: it follows the one category chosen at each level, so one wrong choice near the top is not recovered from.

## Sorting and filtering

`RDM.Filter` sorts what the model rated by its probabilities, likeliest first, and keeps the likeliest: the yeses of yes or no answers, as above, or the options of a `Choice`. `top` is how many go on at most. Here a tool follows the two likeliest categories at every level, not only the one chosen:

```python
likeliest = RDM.Filter(top=2)

def category(listing, call: RDM.Call):
    """Follow the two likeliest categories, and come to every leaf they reach."""
    leaves = []
    for option, probability in likeliest(call.prediction.category):
        leaves += [option] if option not in taxonomy else call.rdm(signature=question(option), listing=listing).observations["category"]
    return leaves

wide = RDM(question("Products"), tools=[category])
wide.set_lm(TypeSafe("jev-latest"))
print("Cat Window Perches" in wide(listing="A padded shelf that suction-cups to a window, so a cat can sun itself.").observations["category"])
# True
```

That is a beam with no score: every leaf the two likeliest reach comes back. To rank whole paths, multiply the `probability` of each step on the way down. `Filter` has no cut of its own. What makes an answer a yes is its predictor's threshold, which belongs to the program (see [Thresholds](#thresholds)).

## With a generative model

A generative model is a tool like any other. It writes; the decision model decides about what it wrote.

```python
class Route(dspy.Signature):
    """Route a request to a smart home."""

    task: str = dspy.InputField()
    house: dict = dspy.InputField()
    split: Noul = dspy.OutputField(desc="Does `inputs.task` ask for more than one distinct thing?")
    act: Noul = dspy.OutputField(desc="Is `inputs.task` one thing to do to the house?")
    chat: Noul = dspy.OutputField(desc="Is `inputs.task` only a question to answer in conversation?")

llm = dspy.LM("openai/gpt-5-mini")
write = dspy.Predict("task -> parts: list[str]")
write.set_lm(llm)
reply = dspy.Predict("task -> reply")
reply.set_lm(llm)

def split(task, house, call: RDM.Call):
    """Each part the generative model writes is a request of its own, routed one level down."""
    return [call.rdm(task=part, house=house) for part in write(task=task).parts]

route = RDM(Route, tools=[split, dspy.Tool(act, name="act"), dspy.Tool(reply, name="chat")])
route.set_lm(TypeSafe("jev-latest"))
result = route(task="Turn the TV off, and tell me whether it will rain.", house=house)
print(house["tv"])
# off
```

A module is a tool when it is wrapped in `dspy.Tool`: it is called with its signature's inputs, taken by name. `act` is the module of the first example, another `RDM`: called as a tool it joins the run as a sub-call, and behaves there as it does alone. Its own limits hold for what it does, its depth is counted from where it was called, and the run's requests and tokens bound them both. Setting the client on `route` leaves the tools' own models alone. What a tool's model uses counts against `max_tokens` and is recorded as `tool_tokens`.

## A REPL: pointing at code a language model wrote

A decision model cannot count, loop or compare, and it can only point at what is on its map. A language model can write. `RDM.Repl` is what the two share: a sandboxed Python REPL, of the kind [`dspy.RLM`](../modules/RLM.md) runs in, where the tools it is given are functions. `repl.define(name, description, code)` defines one function in it. Given to an RDM among its tools, the REPL puts every function written so far on the map, and when the decision model points at one, the REPL runs it.

```python
house["heat"] = 21

def cooler():
    """Make it one degree cooler."""
    house["heat"] -= 1
    return f"heat: now {house['heat']}"

def heat():
    """How warm it is, in degrees."""
    return house["heat"]

repl = RDM.Repl(tools=[cooler, heat])                # the sandbox, and what written code may call in it

class Write(dspy.Signature):
    """Write one Python function that does what the request still needs, and define it with `define`.

    Where the function runs, `cooler()` makes it one degree cooler and `heat()` returns the degrees it is now.
    """

    task: str = dspy.InputField()
    house: dict = dspy.InputField()
    wrote: str = dspy.OutputField(desc="What `define` returned, word for word.")

writer = dspy.RLM(Write, tools=[heat, repl.define])  # what the writer may call while it writes: reading, and defining
writer.set_lm(llm)
write = dspy.Tool(writer, name="write", desc="Have a new tool written in code and put on this list: nothing listed does what is still needed.")

cool = RDM(Operate, tools=[write, repl])
cool.set_lm(TypeSafe("jev-latest"))
cool(task="Bring the heat down to 18 degrees.", house=house)
print(house["heat"])
# 18
```

Nothing here is special to writing. `repl.define` is a plain function, and `write` is a tool like any other: a `dspy.RLM` whose own tools are `heat` and `define`. The decision model points at `write`, the language model writes a function and defines it, the function's name is on the map from the next request, and the decision model points at it. The language model never chooses what happens. It makes something new to point at, with what a decision model lacks: a loop, arithmetic, a condition. It is given only `heat` beside `define`, so it cannot act while it writes. Give it `repl.show` as well and it can read what is written already, with the names it cannot take, so that it writes nothing twice.

Writing only defines: the source must be one `def` and nothing beside it, under a name no tool has, and anything else is refused with the reason. A function that fails when it is run is taken off the map, and the error is the observation the next request shows, so it can be written again. The request's inputs are variables in the REPL as copies, so a function changes the world through the tools. A REPL runs one thing at a time: one written function calls another directly, but code it is running cannot call back into it through a tool.

What a program finds out is kept the same way. There is no separate place for notes: a note is a function that returns what was found, and pointing at it reads it.

```python
repl.define("boiler", "What was found wrong with the boiler.", "def boiler():\n    return 'The pilot light is out, and the gas valve is open.'")
print(repl.execute("print(boiler())"))
# The pilot light is out, and the gas valve is open.
```

As in a `dspy.RLM`, the REPL is where the program keeps what it made, and the map is how the decision model reaches it: `boiler` is now a name on `cool`'s map beside the function written above. With `RDM.Near` it has labels as any name has, and is offered when a request is near it.

A written function is a tool like any other. A parameter it takes is filled by name from the request's answers, then its inputs, so what a function should be told can be one more question of the request: with an output `degrees: Literal[1, 2, 3]` beside `next`, one `cool_by(degrees)` serves every number, and the decision model chooses the number.

So an RDM runs in one of three ways, by what its map can hold:

- **No REPL.** The map holds what was written ahead of time, and what code keeps, as `good night` above.
- **One REPL.** What is written stays, for the rest of the run and the runs after it. `repl.written` holds each function's description and source. It is part of the program: `save` writes it, `load` defines the functions again, and a copy of the program has a REPL of its own with the same functions.
- **Several REPLs.** `repl.spawn(name, description)` opens another environment, isolated from the first: a function written in one does not exist in another. A name on a map can stand for a whole environment, so pointing at it routes the request there:

```python
def room(name, description):
    """An environment, a tool that writes into it, and a module over the two: one name on a map."""
    environment = repl.spawn(name, description)
    writes = dspy.RLM(Write, tools=[heat, environment.define])
    writes.set_lm(llm)
    return dspy.Tool(RDM(Operate, tools=[dspy.Tool(writes, name="write", desc=write.desc), environment]), name=name, desc=description)

home = RDM(Operate, tools=[room("heating", "Anything about the temperature."), room("lights", "Anything about the lights.")])
repl.shutdown()  # shuts the interpreters down, those of the environments it opened included
```

The interpreter comes from `interpreter_factory`, as for `dspy.RLM`: `dspy.PythonInterpreter`, the Deno sandbox, unless the argument or `dspy.configure(interpreter_factory=...)` names another, such as Monty. One that starts quickly suits many environments.

## Inputs of any size

`fit=` takes what is given each request before it is asked and returns the signature and inputs to ask in its place. `RDM.Fit` is the one about size: it cuts down what one request cannot hold, deciding with the same model what to keep: a long input to the parts the questions need, and a `Choice` with too many options to a shortlist.

```python
covered = RDM(Search, fit=RDM.Fit(max_chars=400))  # no request shows more than 400 characters of the text
covered.set_lm(TypeSafe("jev-latest"))
print(covered(text=zen, question="How many obvious ways should there be to do something?").look.value)
# True
```

Each request `Fit` makes is an answers-only sub-call, so it counts against the run's limits and is in the trajectory. Reading an input in parts costs requests in proportion to its length, and a shortlist suits a question whose best answer is the best of the groups' best.

## A map with labels

On a long map the decision model reads every description each time it points. `RDM.Near` gives each name on the map labels as well: its answers to the questions of a signature, each question a dimension with a name. A name is asked once. Each request is asked the same questions, and the pointer is offered the names nearest it.

```python
class About(dspy.Signature):
    """Say what a text is about."""

    task: str = dspy.InputField()
    screen: Noul = dspy.OutputField(desc="Is `inputs.task` about the TV?")
    light: Noul = dspy.OutputField(desc="Is `inputs.task` about the lights?")
    entry: Noul = dspy.OutputField(desc="Is `inputs.task` about the door or its lock?")

near = RDM.Near(About, top=2)
guided = RDM(Operate, tools=steps, fit=near)
guided.set_lm(TypeSafe("jev-latest"))

house.update(door="open", lock="unlocked")
guided(task="Lock the front door.", house=house)
print(house["door"], house["lock"])
# closed locked
```

The six steps are each asked the three questions once, as `"lock locked: Set the lock to locked."`, and answer something like `{"screen": 0.0, "light": 0.0, "entry": 0.83}`. The request is asked them too, and of the six only `door closed` and `lock locked` are offered, with `finish`. What was answered is kept in `near.answers`, as part of the program: `save` writes it, and a request then costs one question more however long the map is. Every number has a name, so it can be read why a step was offered.

`Near` is about the map: it leaves any other `Choice` to `Fit`. The two work together, given as a list and applied in order: `fit=[RDM.Near(About), RDM.Fit()]` offers the nearest names and then makes the request fit. `finish` always stays on offer, and so do the names given to the module as `always=[...]`, a tool that writes, say. What `Near` asks are sub-calls of the run, as `Fit`'s are, so they count against its limits; label a long map before the run with `near.embed(texts)`. Names are compared by what their labels share, the dot product, unless `measure=` names another way (`"cosine"`, `"euclidean"`, `"manhattan"`, `"jaccard"`) or is a function of your own. Used alone, `near.embed(texts)` and `near.score` are a search over documents by named dimensions.

A dimension can also be a `Score`, which is how far up its rubric a text is, or a `Choice`, whose options are a dimension each. `RDM.Near(About, use=True)` asks one thing more of each request: how it uses each dimension, as a filter, so that only names that have it are offered, as a boost, or not at all; a filter that would leave no name is let go. For documents, `near.embedder()` is the same coordinates as a `dspy.Embedder`, so `dspy.retrievers.Embeddings(corpus=texts, embedder=near.embedder())` searches by named dimensions. Labels are one model's answers, and are dropped when another model is asked.

A dimension is a name and a question, and both can be written while the module runs:

```python
near.add("power", "Is `inputs.task` about switching something on or off?")
print(list(near.questions))
# ['screen', 'light', 'entry', 'power']
```

`near.add` and `near.remove` are plain functions, as `repl.define` is, so a language model writes dimensions as it writes anything else: as the tools of a `dspy.RLM`, `dspy.RLM("task -> made", tools=[near.add, near.remove])`, which can itself be a tool on the map for the decision model to point at. The language model coins the label, its name and its question. The decision model then says of every name on the map, and of every request, whether the label applies. Over a long task this is a place for what a program has learned to tell apart, kept outside any request and readable by a person.

`Near` has no threshold or weight of its own. A label is what the model answered, until its dimension is given a parameter on `near.dimensions`, the predictor that asks: a threshold, under which a yes or no counts as nothing; cuts, which give a score its level; weights, which multiply a choice's options. Those are fitted by `ReAnchor` like any other (see [Thresholds](#thresholds)), and they are read when a label is, so changing one asks nothing again.

## The helpers at a glance

| Helper | Where it goes | What it is for |
|--------|---------------|----------------|
| `RDM.Fit()` | `fit=` | Size: a long input, or a `Choice` with too many options, cut to what one request holds |
| `RDM.Near(dimensions)` | `fit=` | A long map: labels for its names, and the nearest offered |
| `RDM.Repl(tools=...)` | among `tools` | What a language model writes, in a sandbox: functions to run, and what was found out, kept as functions that return it |
| `RDM.Filter(top=...)` | called in a tool of yours | Sorting what the model rated |

`fit=` takes what changes a request before it is asked, one or a list applied in order. What goes among the `tools` puts names on the map. A language model writes into a `Repl` or a `Near` the same way, through plain functions (`repl.define`, `near.add`), and each has `show()` for the writer to read what is there. What they hold is part of the program, and each change made during a run is recorded in the trajectory, under `changes` on the request whose tool made it.

## How it works

1. **One request, answered once.** A call asks the signature's questions once, as `dspy.Predict` would. With no tools, that is all it does.
2. **An answer runs a tool.** An output named after a tool calls it on a yes (`Noul`) or a chosen option (`Choice`, where `None` calls nothing). An output typed `dspy.Tool` points at any of the module's tools, or at `finish`. Every tool selected runs, at once.
3. **A tool's arguments are taken by name** from the request's answers, then its inputs, and a parameter annotated `RDM.Call` receives the request. So an argument the model should choose is one more question of the same request. A required parameter that names neither is an error.
4. **What a tool returns is its observation.** Calling `call.rdm` inside a tool is a sub-call, one level down. After a tool an output pointed at, the same question is asked again.
5. **A run stops** when no answer selects a tool, at `finish`, at `max_depth` (where tools are not run), when a request comes round that was already asked on the way to it (`repeat`), or at a limit.

A call returns a `Prediction` with the answers its own request gave, so a run stands wherever a request does: as a tool of another module, or as a sub-call of itself.

## Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| Signature` | required | The questions. An output named after a tool calls it; an output typed `dspy.Tool` points at any tool. |
| `tools` | `list[Callable \| dspy.Tool]` | `None` | Functions, modules, other RDMs. Kept in `rdm.tools`, which can change while the module runs. Something that offers tools of its own, an `RDM.Repl`, is given whole. |
| `always` | `list[str]` | `None` | Names of tools that stay on offer whatever a `fit` cuts from the map, as `finish` does |
| `fit` | `Callable \| list[Callable]` | `None` | Given each request as `fit(signature, inputs, call)`; returns the signature and inputs to ask instead. Several are applied in order. |
| `finish` | `bool` | `True` | Whether an output typed `dspy.Tool` is offered `finish` beside the tools |
| `max_iters` | `int` | `20` | Turns one pointer's loop may take |
| `max_depth` | `int` | `3` | The depth at which tools are not run. The module's first call is at depth 0, wherever it was called from. |
| `max_predictor_calls` | `int` | `256` | Requests answered in one run, sub-calls included. A module called inside another's tool keeps its own limits there as well. |
| `max_tokens` | `int` | `None` | Tokens one run may use, its tools' models included |

## Tools

| A parameter | Receives |
|-------------|----------|
| Named after an output of the request | That answer's value |
| Named after an input of the request | That input, whole |
| Annotated `RDM.Call` | The request: its `prediction` with probabilities, its `inputs`, the `state` (what was passed as `rdm(..., state=...)` to this call or the nearest above it), the run's `tools`, the `trajectory` so far, `depth`, `iteration`, `requests_left`, and `rdm`, the module that is running |

A tool selected by an output has that output's name, so it takes the answer that selected it by the same name: `def category(category, ...)`. A required parameter that names no answer or input is an error.

A tool is called as a `dspy.Tool`, so tool callbacks fire and an async tool runs under `acall`. What a tool returns is its observation, and an input named `observation` shows it as JSON when it is not a text. A tool that raises is an observation too, as in `ReAct`, and not the end of the run. Using the module wrongly, in a tool or not (a parameter that names nothing, inputs that do not match the signature), raises.

## Output

| Field | Description |
|-------|-------------|
| The signature's outputs | The answers this call's request gave |
| `observations` | What each selected tool returned; an output named after a tool that selected nothing has no entry. For an output typed `dspy.Tool`, every tool it pointed at in turn, each with what it returned, ending with `["finish", None]` when it pointed at `finish`. |
| `termination_reason` | `final`, or what stopped something in this call or below it: `repeat`, `max_depth`, `max_iters`, `max_predictor_calls` or `max_tokens` |
| `trajectory` | On the call that started the run: every request in order, with the `changes` its tools made to a `Repl` or a `Near`, with its `depth`, `iteration`, `parent`, `tool_name`, the `inputs` it showed, the `outputs` it got with their probabilities, the `model` that answered and the `tokens` it used. An entry with a `model` was put to the model; one marked `repeat` or `skipped`, or whose outputs are all `unasked`, was not. |

## Thresholds

Every cut the module makes is a parameter of a predictor the program holds, so it is saved with the program and [`ReAnchor`](ReAnchor.md) can fit it against a metric. There are four places:

```python
search.predict.fields["look"] = {"threshold": 0.7}          # a question of the module's own signature

lines.search = dspy.Predict(Search)                         # a question asked only in sub-calls: give the module
lines.search.fields["look"] = {"threshold": 0.7}            # a predictor for it, and its requests are answered by it

covered.fit.needed.fields["needed"] = {"threshold": 0.3}    # Fit's question about each part of a long input (`fit[0]` where `fit` is a list)

guided.fit.dimensions.fields["entry"] = {"threshold": 0.7}  # Near's question about a dimension
```

Without a predictor of its own, a sub-call's question is answered by the module's, with any threshold `rdm.predict.fields` holds for an output of that name; that works, but it is not saved or fitted. `Fit` keeps the likeliest parts that fit until `needed` is given a threshold; then it keeps only the parts said to be needed.

## Notes

!!! note "Thread safety"
    The tools one request selects run on up to `dspy.settings.num_threads` threads, and so do the sub-calls of one `call.rdm.batch`. A tool that keeps state of its own must be safe to call concurrently. Sub-calls made on threads a tool starts itself are separate runs; `call.rdm.batch(...)` keeps them in the run.

!!! note "Option order and model versions"
    A decision model can lean towards the options listed first. A pointer offers the tools in the order of `rdm.tools`, with `finish` last. DSPy's cache key sorts a `Choice`'s options, so the same options in another order are one cache entry: use `TypeSafe(..., cache=False)` to measure how the order of a map's names matters. A threshold is fitted to one model, so name a version of the model in place of `jev-latest` once thresholds are fitted.

!!! note "Saving and copying"
    `save`, `set_lm`, `named_predictors` and the optimizers reach the predictors the module holds: `predict`, `fit.needed`, and any predictor set on it. A module wrapped as a tool is not reached through the map, and neither is a `Repl`'s writer: hold them in a module of your own beside the RDM (`self.act = ...`, `self.repl = ...`) and they are saved and optimized with it. What a helper keeps is part of the program: a `Repl`'s functions, and a `Near`'s dimensions and labels, are written by `save` and read by `load`, and a copy of the program has its own, with a REPL of its own. `save(..., save_program=True)` saves a program that holds a `Repl` too; the REPL starts again when it is first used. A tool put in `rdm.tools` by code is code, and is not saved. A `Score` or `Choice` dimension comes from the signature a `Near` is built with; a dimension added while it ran is saved as the yes or no question it is.

## API Reference

<!-- START_API_REF -->
::: dspy.experimental.RDM
    handler: python
    options:
        members:
            - __init__
            - __call__
            - forward
            - aforward
            - batch
            - deepcopy
            - dump_state
            - get_lm
            - load
            - load_state
            - named_parameters
            - named_predictors
            - parameters
            - predictors
            - reset_copy
            - save
            - set_lm
        show_source: true
        show_root_heading: true
        heading_level: 3
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

### `RDM.Call`

::: dspy.predict.rdm.Call
    handler: python
    options:
        show_source: false
        show_root_heading: false
        heading_level: 4
        docstring_style: google

### `RDM.Fit`

::: dspy.predict.rdm_fit.Fit
    handler: python
    options:
        members:
            - __init__
            - __call__
        show_source: true
        show_root_heading: false
        heading_level: 4
        docstring_style: google

### `RDM.Filter`

::: dspy.predict.rdm_filter.Filter
    handler: python
    options:
        members:
            - __init__
            - __call__
        show_source: true
        show_root_heading: false
        heading_level: 4
        docstring_style: google

### `RDM.Near`

::: dspy.predict.rdm_near.Near
    handler: python
    options:
        members:
            - __init__
            - add
            - remove
            - show
            - embed
            - embedder
            - score
        show_source: true
        show_root_heading: false
        heading_level: 4
        docstring_style: google

### `RDM.Repl`

::: dspy.predict.rdm_repl.Repl
    handler: python
    options:
        members:
            - __init__
            - define
            - remove
            - show
            - offers
            - execute
            - spawn
            - shutdown
        show_source: true
        show_root_heading: false
        heading_level: 4
        docstring_style: google
