# dspy.CLI

<!-- START_API_REF -->
::: dspy.CLI
    handler: python
    options:
        members:
            - __call__
            - acall
            - aforward
            - dump_state
            - forward
            - load_state
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->

## Command presets

`dspy.resolve_agent_cli()` provides small, typed conveniences for documented
one-shot agent commands. Its result is a `dspy.CLICommand`, which can be passed
directly to `dspy.CLI`. Use `prefix_args` for global options that must precede
the preset subcommand and `extra_args` for options that follow it.

```python
command = dspy.resolve_agent_cli(
    "codex",
    prefix_args=("-a", "never"),
    extra_args=("--sandbox", "read-only"),
)
lm = dspy.CLI(command)
```

Presets do not select approval, sandbox, permission, or session policy. Pass
such policy explicitly, or use an explicit argument vector instead.

## Adapter retries

`CLI` is an ordinary DSPy language model, so it follows the configured
adapter's retry policy. In particular, the default `ChatAdapter` retries once
with `JSONAdapter` when a command exits successfully but its stdout cannot be
parsed. For a stateful command that must run at most once, disable that fallback:

```python
dspy.configure(
    lm=dspy.CLI(command),
    adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
)
```

Transport failures such as a nonzero exit are not retried by the adapter.
