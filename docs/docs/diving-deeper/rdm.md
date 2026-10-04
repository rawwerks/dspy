# RDM: recursive decisions

## Intent

`RDM`, or “Recursive Decision Model”, is the idea of `dspy.RLM` for a model that returns decisions instead of text. A System One model such as Jev is given a state and typed questions, and answers each with a decision and a probability. It writes nothing, so it cannot drive a REPL, and it never makes a state. What it can do is decide, at every node of a run, what makes the next state: code, a generative model, or another module. `dspy.experimental.RDM` is that loop and nothing else: ask a request once, let the answers select tools or point at one, and decide again about whatever the tools made. Try it when a decision should determine what is done or asked next.

The approach is described in ["Recursive Decision Models" (Weitekamp, 2026)](https://raw.works/recursive-decision-models/). The idea it follows is that of ["Recursive Language Models" (Zhang, Kraska, Khattab, 2025)](https://arxiv.org/abs/2512.24601).

## Design decisions

### 1. A request is a DSPy signature: state in, typed decisions out

One request is one `Predict` call over a signature whose outputs are decisions (`Noul`, `Choice`, `Score`, or `bool` and `Literal`). That is DSPy's spelling of a System One request: the inputs are the state, the output fields are the questions, and the prediction holds each decision with its probabilities. All of a request's questions are answered together and independently, including ones that turn out not to matter.

### 2. When no answer selects a tool, it is `Predict`

`dspy.experimental.RDM(signature)` takes the signature's inputs and returns its outputs. With no tools, or when no answer selects one, that is one request and the answers `dspy.Predict(signature)` gives. A whole run returns the same thing, so a run can stand wherever a request does: as a tool of another module, or as a sub-call of itself.

### 3. A request is asked once

Nothing is asked twice. A judgment is made on the state it was given and stands as made. What is asked next is always a different request, built for that purpose with only the state its questions need. A `Choice` with one option leaves nothing to decide: it is answered with that option, as certain, and not asked.

### 4. An answer selects a tool by name

An output named after a tool calls that tool. A yes calls it, and so does a chosen option, which is passed to it; choosing `None` calls nothing, and a `Score` never does. Every tool that is selected runs, at once. Any other output is a plain judgment. To select several of some options built at run time, ask one `Noul` for each beside one `Noul` named after the tool: the model's probabilities for separate plain questions put things in a useful order far more reliably than its yes or no does.

### 5. An output typed as a tool points: the map

A decision model cannot write a tool's name, but it can point at one. An output typed `dspy.Tool` is a `Choice` over the module's tools, built when each request is made from their names and descriptions, with `finish`. It is ReAct's `next_tool_name` for a model that points. The tool pointed at runs, and the same question is put again, showing the inputs as they are now and, in an input named `observation`, what the tool returned, until `finish`. So the tools are a map: names with a description for the model and a thing for the code to do. The map is `rdm.tools`, and each run points at its own copy of it, so runs on other threads do not see each other's changes. A tool put in the run's copy, `call.tools`, by another tool or by a generative model inside one, is on offer from the run's next request; one put in `rdm.tools` is on offer from the next run. That is how a run keeps what it did as one more name. The loop is the module's and not the model's: each turn is one closed question about the state as it now is, and code bounds the turns. They are a loop at one level, so a loop is as long as `max_iters` allows, and pointing at `finish` is recorded as `["finish", None]`. Where one request can decide, one request is better: the loop is for when what a step does changes what the next should be. `finish=False` leaves it out, where stopping is not the model's to choose. A map's names can be another `dspy.experimental.RDM`, so pointing at one asks a question one level down.

### 6. A tool does, and decides nothing

A tool is a function, a generative module such as `dspy.Predict` on an LLM, a `dspy.RLM`, or another `dspy.experimental.RDM`. What it returns is its observation, whatever its shape: the module never reads a returned value as an instruction. So the decision model is the only thing that chooses, at the root and at every node below it, and a tool is the only thing that makes a state: by slicing what is there, by acting, or by generating. Pair a decision model with a generative one this way and the two run as a machine: one widens a state into several, the other decides about each. With code as the only tools it runs with no generative model at all.

### 7. A tool recurses by calling the module

A tool that takes a parameter annotated `RDM.Call` is given the request that selected it, and `call.rdm` is the module that is running. Calling it is a sub-call: it joins the run in progress, one level deeper, and counts against the same limits. Its prediction comes back to the tool, which can count, filter or sum up what it got, and ask again about that. `signature=` puts other questions for the sub-call, so each answer can decide what the next question is. `max_depth=0` asks for the answers only: no tool is run for that call. It is the module's counterpart of an RLM's plain `llm_query` beside its recursive call. `call.rdm.batch(...)` makes many sub-calls at once. Recurse through `call.rdm` rather than a variable that holds the module: an optimizer runs a copy, and the copy's own thresholds should apply below.

### 8. A call returns its own answers

The answers a call returns are the ones its request gave, as `dspy.Predict` would return them. What its tools came to, sub-calls included, is in `observations`. A judgment made on less is never overwritten by one made on more: a later question about what came back is asked under a name of its own, in the request that asks it.

### 9. The state is outside every request

The inputs of a run, anything passed to it as `state=`, and the history of every request made for it are the state. The model never sees it whole: a request shows what it was given and no more. `state=` is the place for what no request should show: an index, a corpus, a game, an identifier. A tool reads it on its `RDM.Call`, as `call.state`, beside the `trajectory` so far, so its names never collide with a signature's. State given to a call is that call's and that of the calls under it, so a module that is given state works the same when another module calls it as a tool.

### 10. Arguments are chosen or taken, never written

A decision model does not write an argument. It can choose one: each parameter of a tool is filled by name from the request's answers, so an argument from a closed set is one more question of the same request, a `Literal` or a `Choice` beside the question that selects the tool. What no answer names is taken from the request's inputs, given whole. A tool selected by an output has the output's name, so the answer that selected it comes by that name too. A required parameter that names nothing is an error: a misspelt name never quietly receives something else. A tool that needs the probabilities behind an answer reads `call.prediction`.

### 11. The module knows no task, and no size

There is no built-in tool, no fixed question and no store. There is also nothing about how large a request may be. `fit` is the one place for that: a callable given each request before it is asked, which returns the signature and inputs to ask in their place, and can make sub-calls as a tool can. `RDM.Fit` is one, a `dspy.Module` written only with what any tool can use (see below).

### 12. A run stops at a decision, at a repeat, or at a limit

A call ends when no answer selects a tool, or when an output typed as a tool points at `finish`. It also ends when a request comes round that was already asked on the way to it, in a turn before it or in a call above: the same questions about the same state would get the same answers and select the same tools, so it is not asked, and the call returns the answers it got before with `termination_reason` `repeat`. A request is taken for the whole of what matters: when something no request shows has changed (time has passed, a page was turned), show it in the request, or the same request will be taken for a repeat. And it ends at a limit: `max_depth` is where tools stop being run, `max_iters` is the turns one pointer's loop may take, `max_predictor_calls` and `max_tokens` bound the whole run, the tokens a tool's own model uses included. Whatever stopped something other than a decision is the `termination_reason` of every call above it, so a run that reports `final` left nothing out. When several things did, the reason is the one that bounds the most: a limit of the run, then a limit of a call, then a repeat. Another `dspy.experimental.RDM` called from a tool joins the run and behaves there as it does alone: its own limits hold for what it does, its depth is counted from where it was called, and a request it asks is a repeat only of its own. The run's requests and tokens bound them both.

### 13. A tool is called as DSPy calls one, and one that fails is an observation

Every tool is a `dspy.Tool` and is called as one, so tool callbacks fire, and an async tool runs under `acall`. As in ReAct, an exception raised by a tool is returned to the caller as a message in place of a result.

### 14. Every cut is a parameter of a predictor

A free parameter must be one an optimizer can fit. So nothing in the module cuts by a number of its own: whether a `Noul` is a yes is its predictor's threshold, `fields[name]["threshold"]`, as for any decision output. A request is answered by the predictor the program holds that declares its questions: the module's own `predict`; `fit.needed`, the question `Fit` puts to each part of a long input; or a predictor given to the module for the questions of a sub-call (`rdm.judge = dspy.Predict(Judge)`). Those parameters are named, saved with the program, and fitted by `ReAnchor`, which reruns the whole program against the metric. A question no predictor declares is answered by the module's own, with any threshold `rdm.predict.fields` holds for an output of that name; parameters fitted to one answer space, such as a `Choice`'s weights, are not applied to a request that asks the same name over other options. `Fit` fills the room with the likeliest parts until `needed` has a threshold, `Filter` sorts and keeps the yeses without a cut of its own, and `Near` measures by the dimensions its predictor says yes to, with no weights of its own.

### 15. A language model writes with plain functions, and what it writes is the program's

A map holds only what someone wrote. Two helpers let a language model write more of it while the module runs, and both do it the same way: with plain functions. `repl.define(name, description, code)` defines a function in a sandboxed REPL, and `near.add(name, question)` adds a label. There is no third for notes. As in a `dspy.RLM`, the REPL is where a program keeps what it made: something found out is a function that returns it, defined like any other, and pointing at it reads it. Each has `show()`, which is what a writer reads first: what is there already. Neither is a protocol of the module. They are the tools of a `dspy.RLM`, or what a function of yours calls, and the module that writes is a tool on the map like any other, which the decision model points at. The language model still chooses nothing: it makes something to point at, and the decision model points.

What is written is state of the program, in the way DSPy holds state: a parameter the module reaches. A `Repl` given among the `tools` is part of the module, and a `Near` given as `fit` is too, so `save` writes the functions and the labels, `load` reads them, and a copy of the program, as an optimizer makes, has its own. Each change made in a run is recorded in the trajectory, under `changes` on the request whose tool made it: what was defined, added, removed, or failed. A `Repl` owns its interpreter. `dspy.RLM` makes a fresh interpreter for each call and shuts it down, and its docs make that the contract of an `interpreter_factory`; so a REPL kept alive after an RLM returns would break it. A `Repl` takes the same `interpreter_factory`, makes one interpreter, and runs each written function with the inputs of the request that pointed at it as variables.

Writing only defines. The source `define` takes is one `def` and nothing beside it, with no decorator and plain defaults, under a name no tool has, so nothing runs when a function is written; and a writer given only tools that read cannot act while it writes. A writer that could would have chosen something.

### 16. RDM is marked experimental and the interface is in flux

The class carries the `@experimental` decorator. Treat the API as subject to change between releases and pin a version if you depend on it.

## Fit: a request too large for one

`dspy.experimental.RDM(signature, fit=RDM.Fit())` answers what `dspy.Predict(signature)` answers at any size. `Fit` is given each request before it is asked and cuts down what one request cannot hold, by sub-calls that are asked for their answers only:

- **An input longer than `max_chars`** is split into consecutive parts: a text between lines, a list between items, a dict between entries. Each part is asked whether it helps to answer the questions. The likeliest parts are kept, for as long as there is room, and at least one: a fixed cut means something different on each model. The question is a predictor, `fit.needed`: once it has a threshold, set by hand or fitted by `ReAnchor`, only the parts it says are needed are kept.
- **A `Choice` with more than `max_options` options**, or with options longer than a request, is put to the options a group at a time. Each group's likeliest go on, two where a group has them, until what is left fits. Groups asked about the same inputs are questions of one request, so the inputs are sent once. An option of `None` is offered to every group.
- **When the options are the keys of a dict input**, each option points at one entry, and the two are cut down together: a group is some of the entries with exactly their keys as its options, so the question is put to every entry.

Every size is an argument of `Fit`, since each suits a model and not a task: `max_chars` and `max_options`, how many groups one request asks about (`max_questions`), how many parts of a long input a request has room for (`parts`), and how many of a group's options go on (`keep`). The defaults are the ones tried on Jev.

Cutting a `Choice` down suits a question whose best answer overall is the best of the groups' best, and not a question about the options as a set. Taking an input in parts reads all of it once, so the cost grows with the input; what should not be read at all belongs in `state=`.

## Near: a map with labels

A name on a map has one dimension, its description. `RDM.Near(signature)` gives it more: the signature's outputs are yes or no questions, each a dimension with a name, and a name's labels are its answers to them. This is an embedding whose every number can be read, made by asking questions and not by an embedding model.

- **A name is asked once**, as `"name: description"`, and what it answered is kept: the probability of a yes for each dimension, as the model gave it. The labels are part of the program, saved and copied with it. Only a text that lacks an answer is asked.
- **A request is asked the same questions**, about its input of the same name as the signature's one input. Names are ranked by how near their labels are to the request's, and the pointer's options are cut to the `top` nearest, in the order they had. A name that shares nothing with the request is not offered; `finish` and the names the module always offers are.
- **A request with no direction is asked as it is.** When it shares nothing with any name, cutting would be a guess.
- **A dimension is a name and a question, and a language model can write both**, with `near.add` and `near.remove` (decision 15). A new dimension asks each name once more, when next it is needed.
- **A dimension is a yes or no, a score, or a choice.** A `Noul` is the probability of a yes. A `Score` is how far up its rubric the text is. A `Choice` is a dimension for each option but "none of these", as likely as the option was.
- **A request can say how it uses each dimension.** With `use=True` it is asked, of every dimension, whether it is a filter, a boost, or to be ignored. A dimension counts as far as it is not ignored, and where it is a filter only names that share it are offered. A filter that would leave no name is let go: a hard cut made on one answer is the mistake to avoid.
- **Labels are one model's answers.** They are kept with the name of the model that gave them, and dropped when another is asked.

`Near` is about the map, and `Fit` about size. `Near` cuts only the output that points at a tool; a `Choice` that is not the map is left to `Fit`. Given together as a list, they are applied in order: the nearest names first, then whatever it takes for the request to fit. Both leave on offer what the module always offers (`always=`), so they cannot disagree about it.

`Near` holds no weight and no cut. A label is what the model answered, until its dimension has a parameter: a threshold, under which a yes or no counts as nothing; cuts, which give a score its level; weights, which multiply a choice's options, and so are a weight for each dimension it makes. They are read from `near.dimensions`, the predictor that asks, each time a label is read (and those of how a request uses a dimension from `near.uses`): parameters of the program, which `ReAnchor` fits against the program's metric with no change to `ReAnchor`, and changing one asks nothing again. So that `ReAnchor` sees the labels it did not ask for during a run, `Near` records each kept answer it reads as evidence that predictor decoded. The measure is an argument: the dot product by default, `"cosine"`, `"euclidean"`, `"manhattan"`, `"jaccard"`, or a function of two sets of coordinates. What `Near` asks in a run are sub-calls for the answers only, as `Fit`'s are, so they count against the run's limits and are in its trajectory; a long map is labelled before the run with `near.embed(texts)`. For documents, `near.embedder()` gives the same coordinates as a `dspy.Embedder`, so that the search is DSPy's own.

## Repl: three ways to run

An RDM runs in one of three ways, by what its map can hold:

- **No REPL.** The map holds what was written ahead of time, and what code or a tool keeps during a run.
- **One REPL.** `RDM(signature, tools=[*steps, write, repl])`, where `write` is a tool of yours that calls `repl.define`. Each request's map holds the functions the REPL offers then, so what is written is on the map from the next request, and stays for the runs after it. A function that fails is taken off the map, and its error is the next request's observation, so it can be written again; its source is kept, marked as failed.
- **Several REPLs.** `repl.spawn(name, description)` opens another environment from the same factory. A function written in one does not exist in another, nor does anything else it holds. A module over one environment, wrapped as a tool under the environment's name, makes the name stand for the whole environment on another map, so pointing at it routes a request there.

An interpreter belongs to one thread, and a module's tools run on several. So each `Repl` runs everything on one thread of its own, one call at a time, whichever thread asks. A REPL runs one thing at a time, so code it is running cannot call back into it through a tool: that is refused, and said, on every interpreter. One written function calls another directly, inside the REPL, and code in one environment can reach a function written in another, which has an interpreter of its own.

## API walkthrough

### Defining and running an RDM

**`dspy.experimental.RDM(signature, tools=None, fit=None, always=None, finish=True, max_iters=20, max_depth=3, max_predictor_calls=256, max_tokens=None)`**
The constructor builds one predictor, `predict`, over the signature, and keeps the tools by name in `tools`, which can change while the module runs. Something given among the tools that offers tools of its own, an `RDM.Repl`, is kept in `sources`, and each request's map holds what it offers then. `always` names the tools that stay on offer whatever a `fit` cuts. The field names `signature`, `state`, `max_depth`, `observations`, `trajectory` and `termination_reason` are reserved.

**`__call__(signature=None, state=None, max_depth=None, **inputs)` / `acall(...)`**
The call answers the request and runs the tools its answers select. Outside a run it starts one; inside one of its own tools it is a sub-call. `signature=` puts other questions for this call, as it does on `Predict`. `state=` gives this call, and the calls under it, state that tools can read and no request shows. `max_depth=` lowers how far below this call tools may run; 0 asks for the answers only. `acall()` runs the same calls in a worker thread.

### Writing a tool

**Parameters**
Named after an output of the request: that answer's value. Named after an input of the request: that input, whole. Annotated `RDM.Call`: the request, with its `prediction` (the answers, with their probabilities), its `inputs`, the `state`, the run's `tools`, the `trajectory` so far, its `depth`, `iteration` and `requests_left`, and `rdm`, the module that is running. A required parameter that names none of these is an error.

**Return value**
Whatever a tool returns is its observation. After an output typed as a tool pointed at it, it is also shown as `observation` when the same question is put again.

**Sub-calls**
Call `call.rdm`: `call.rdm(record=text, question=question)` for the module's own questions, or `call.rdm(signature=Judge, ...)` for others. For many at once use `call.rdm.batch(...)`: inside a run they are admitted in the order given.

**A module as a tool**
Wrap it with `dspy.Tool(module, name=...)`. It is called through its signature's inputs, filled the same way. Another `dspy.experimental.RDM` wrapped this way joins the run as a sub-call. What a tool's own model uses is recorded as `tool_tokens` and counts against `max_tokens` once the tool returns. What the requests it asks through the module use is counted on those requests, and not again on the tool.

### Sharing a REPL

**`RDM.Repl(tools=None, interpreter_factory=None, name="repl", description="", written=None)`**
`tools` are the functions that exist inside the REPL for written code to call. Give the `Repl` itself to an RDM among its `tools`: each request's map then holds every function written so far.

**`repl.define(name, description, code)`, `repl.remove(name)`, `repl.show()`, `repl.written`, `repl.spawn(...)`, `repl.execute(code)`, `repl.shutdown()`**
`define` defines one function and puts it on offer; it returns a sentence saying so, and raises with the reason where the source is refused. `remove` takes one away. `written` is each function's description and source, saved with the program. `spawn` opens an isolated environment, kept in `repl.environments`. `execute` runs code in the REPL. `shutdown` ends the interpreter and those of the environments opened from it; a `Repl` is also a context manager.

### Labelling a map

**`RDM.Near(signature, top=8, measure="dot", use=False)`**
`signature` is the dimensions: one input, and a `Noul`, `Score` or `Choice` output for each. Give the `Near` to an RDM as `fit`, alone or in a list with `RDM.Fit()`.

**`near.add(name, question)`, `near.remove(name)`, `near.questions`, `near.answers`, `near.show()`, `near.embed(texts)`, `near.embedder()`, `near.score(a, b)`, `near.dimensions`**
`add` and `remove` write the dimensions. `questions` is each dimension's question, and `answers` each text's probabilities, both saved with the program. `embed` gives texts their coordinates and `score` compares two. `dimensions` is the predictor that asks, which holds each dimension's threshold.

### Inspecting a run

**`Prediction` fields: the signature's outputs, `observations`, `termination_reason`, `trajectory`**
The outputs are the answers this call's request gave. `observations` holds what each selected tool returned; for an output typed as a tool, every tool it pointed at in turn, each with what it returned. `termination_reason` is `final`, or what stopped something in this call or below it other than a decision: `repeat` or a limit. `trajectory`, on the prediction of the call that started the run, lists every request in the order it was made: its `depth`, its `iteration`, the `parent` request and `tool_name` it came from, the `inputs` it showed, the `outputs` it got with their probabilities, the `model` that gave them, the `tokens` it used, and its `observations`; `tool_tokens` where a tool used a model of its own, and `fit_tokens` where `fit` did; `unasked` for outputs that had one option; `repeat` for a request that was not asked again; `skipped` for one a limit stopped.

## Cross-links

- [`dspy.experimental.RDM` API reference](../api/experimental/RDM.md) — parameter table and worked examples.
- [RLM: exploring large contexts with code](rlm.md) — the module whose idea RDM follows.
- ["Recursive Decision Models" (Weitekamp, 2026)](https://raw.works/recursive-decision-models/) — the post this module comes from.
- ["Recursive Language Models" (Zhang, Kraska, Khattab, 2025)](https://arxiv.org/abs/2512.24601) — the paper behind `dspy.RLM`.
- [Decision types and System One models](../api/experimental/DecisionTypes.md) — `Noul`, `Choice`, `Score`, `TypeSafe` and per-field parameters.
- [ReAnchor](../api/experimental/ReAnchor.md) — fitting the thresholds.
