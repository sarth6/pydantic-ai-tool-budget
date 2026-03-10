# pydantic-ai-tool-budget

Per-tool budget reminders for [Pydantic AI](https://ai.pydantic.dev/) agents.

## The Problem

`UsageLimits(tool_calls_limit=N)` is a silent kill switch. The model has no idea it's running low on tool calls until the hard cap fires `UsageLimitExceeded` and the entire run dies — often after the model already did useful work.

This package injects per-tool budget reminders directly into the conversation after each tool result, so the model can see how many calls remain and plan accordingly.

See [pydantic/pydantic-ai#4359](https://github.com/pydantic/pydantic-ai/issues/4359) for the full discussion.

## Install

```bash
uv add pydantic-ai-tool-budget
# or
pip install pydantic-ai-tool-budget
```

Requires `pydantic-ai-slim>=0.100.0`.

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai_tool_budget import budgeted


def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"


def lookup(city: str) -> str:
    """Look up city info."""
    return f"Info about {city}"


agent = Agent(
    'openai:gpt-4o',
    tools=[
        budgeted(search, limit=5),
        budgeted(lookup, limit=3),
        # undecorated tools work normally — no budget tracking
    ],
)
```

After each tool call, the model sees a message like:

```
search: 3/5 calls used, 2 remaining.
```

When a tool's budget is exhausted:

```
search: 5/5 calls used, 0 remaining. This tool's budget is exhausted.
```

## Options

### Threshold — only remind when budget is tight

```python
budgeted(search, limit=10, threshold=3)  # only remind when remaining <= 3
```

### Custom formatter

```python
budgeted(
    search,
    limit=5,
    formatter=lambda name, used, limit: (
        f"Only {limit - used} calls left for {name}. Prioritize."
        if limit - used <= 3
        else None  # suppress when there's plenty of budget
    ),
)
```

## How It Works

`budgeted()` wraps your tool function using `functools.wraps`, preserving its name, docstring, and parameter schema. After each call, it returns a [`ToolReturn`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolReturn) with a `.content` field containing the budget reminder. The framework converts this to a `UserPromptPart` placed after the tool result in the model request — exactly where the model reads it before deciding what to do next.

This means:
- Reminders sit in the **conversation body**, not the system prompt — no prompt cache busting
- Each tool gets its **own** budget tracking — the model sees per-tool counts
- **No string mappings** — you pass the function directly, so typos are `NameError`s
- Works with sync and async tools, with or without `RunContext`

## API

### `budgeted(func, *, limit, threshold=None, formatter=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | *required* | The tool function to wrap |
| `limit` | `int` | *required* | Maximum calls before budget is exhausted |
| `threshold` | `int \| None` | `None` | Only remind when remaining <= threshold |
| `formatter` | `(name, used, limit) -> str \| None` | `None` | Custom reminder formatter |

## License

MIT
