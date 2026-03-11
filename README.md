# pydantic-ai-tool-budget

[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-tool-budget)](https://pypi.org/project/pydantic-ai-tool-budget/)
[![Python](https://img.shields.io/pypi/pyversions/pydantic-ai-tool-budget)](https://pypi.org/project/pydantic-ai-tool-budget/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Per-tool budget reminders for [Pydantic AI](https://ai.pydantic.dev/) agents.**

Give your agent awareness of how many tool calls it has left — per tool, per shared pool, or globally — so it can plan ahead instead of crashing into a hard limit.

## Why?

Pydantic AI's `UsageLimits(tool_calls_limit=N)` is a silent kill switch. The model has no idea it's running low on tool calls until the hard cap fires `UsageLimitExceeded` — often **after** it already did useful work that never gets returned.

`pydantic-ai-tool-budget` fixes this by injecting budget reminders directly into the conversation after each tool result. The model sees exactly how many calls remain and can wrap up gracefully.

```
search: 3/5 calls used, 2 remaining.
```

See [pydantic/pydantic-ai#4359](https://github.com/pydantic/pydantic-ai/issues/4359) for the upstream discussion.

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
    "openai:gpt-4o",
    tools=[
        budgeted(search, limit=5),
        budgeted(lookup, limit=3),
        # undecorated tools work normally — no budget tracking
    ],
)
```

After each call, the model sees a reminder like:

```
search: 3/5 calls used, 2 remaining.
```

When the budget runs out:

```
search: 5/5 calls used, 0 remaining. This tool's budget is exhausted.
```

## Use Cases

### Only remind when budget is tight

Don't clutter the context when there's plenty of budget left. Use `threshold` to only inject reminders when remaining calls drop below a value:

```python
budgeted(search, limit=10, threshold=3)  # silent until ≤ 3 calls remain
```

### Shared budget across tools

Multiple tools can draw from the same pool using `ToolBudget`. This is useful when you don't care *which* tools the agent calls, just that total tool usage stays within bounds:

```python
from pydantic_ai_tool_budget import ToolBudget, budgeted

pool = ToolBudget(limit=20)

agent = Agent(
    "openai:gpt-4o",
    tools=[
        budgeted(search_signals, budget=pool),
        budgeted(get_signal_details, budget=pool),
        budgeted(analyze_competitor, budget=pool),
    ],
)
```

All three tools share the same 20-call budget. The model sees the shared remaining count after every call.

### Exempt tools that shouldn't count

Some tools — like a final "save" or "submit" action — should always be available but still show the shared budget status. Mark them `exempt`:

```python
pool = ToolBudget(limit=20)

agent = Agent(
    "openai:gpt-4o",
    tools=[
        budgeted(search_signals, budget=pool),
        budgeted(get_signal_details, budget=pool),
        # exempt: doesn't count against the pool, but still shows remaining
        budgeted(register_opportunity, budget=pool, exempt=True),
    ],
)
```

`register_opportunity` never decrements the shared counter, but the model still sees "X/20 calls remaining" after calling it.

### Custom behavior when budget is exhausted

Instead of letting the model call a tool that can't do anything useful, intercept it with `on_exhaust`:

```python
budgeted(
    search,
    limit=5,
    on_exhaust=lambda name, used, limit: (
        f"Budget for {name} is exhausted. Summarize what you have."
    ),
)
```

When the budget hits zero, `on_exhaust` is called **instead of** the real tool function. The model gets your message as the tool result and can act on it. If `on_exhaust` returns a `ToolReturn`, it's used as-is; otherwise, the standard budget reminder is appended.

### Custom reminder format

Override the default reminder text entirely:

```python
budgeted(
    search,
    limit=5,
    formatter=lambda name, used, limit: (
        f"⚠️ Only {limit - used} calls left for {name}. Prioritize."
        if limit - used <= 3
        else None  # suppress when there's plenty of budget
    ),
)
```

Return `None` from the formatter to suppress the reminder for that call.

## How It Works

`budgeted()` wraps your tool function using `functools.wraps`, preserving its name, docstring, and parameter schema. After each call, it returns a [`ToolReturn`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolReturn) with a `.content` field containing the budget reminder. Pydantic AI surfaces this as a `UserPromptPart` placed after the tool result in the conversation — exactly where the model reads it before deciding what to do next.

This means:
- Reminders sit in the **conversation body**, not the system prompt — no prompt cache busting
- Each tool gets its **own** budget counter (or shares one via `ToolBudget`)
- **No string mappings** — you pass the function directly, so typos are `NameError`s
- Works with sync and async tools, with or without `RunContext`

## API Reference

### `budgeted(func, *, limit, budget, exempt, threshold, formatter, on_exhaust)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | *required* | The tool function to wrap |
| `limit` | `int \| None` | `None` | Per-tool call limit. Mutually exclusive with `budget` |
| `budget` | `ToolBudget \| None` | `None` | Shared budget pool. Mutually exclusive with `limit` |
| `exempt` | `bool` | `False` | Don't count against shared `budget`, but still show reminders. Only valid with `budget` |
| `threshold` | `int \| None` | `None` | Only remind when remaining ≤ threshold |
| `formatter` | `(name, used, limit) → str \| None` | `None` | Custom reminder text. Return `None` to suppress |
| `on_exhaust` | `(name, used, limit) → Any` | `None` | Called instead of the tool when budget is exhausted |

### `ToolBudget(limit)`

A shared call-count pool. Pass to `budgeted(..., budget=pool)` so multiple tools draw from the same budget.

| Property / Method | Description |
|-------------------|-------------|
| `used` | Number of calls made so far |
| `remaining` | Calls left before exhaustion |
| `is_exhausted()` | Whether the budget is fully consumed |
| `reset()` | Reset the counter to zero |

## License

MIT
