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

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_tool_budget import ToolBudgetToolset

toolset = FunctionToolset()

@toolset.tool_plain
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

@toolset.tool_plain
def lookup(city: str) -> str:
    """Look up city info."""
    return f"Info about {city}"

# Wrap with per-tool budget tracking
agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        ToolBudgetToolset(
            wrapped=toolset,
            limits={'search': 5, 'lookup': 3},
        ),
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

### Default limit for all tools

```python
ToolBudgetToolset(
    wrapped=toolset,
    default_limit=10,  # applies to any tool not in `limits`
)
```

### Threshold — only remind when budget is tight

```python
ToolBudgetToolset(
    wrapped=toolset,
    limits={'search': 10},
    threshold=3,  # only remind when remaining <= 3
)
```

### Custom formatter

```python
ToolBudgetToolset(
    wrapped=toolset,
    default_limit=5,
    formatter=lambda name, used, limit: (
        f"Only {limit - used} calls left for {name}. Prioritize."
        if limit - used <= 3
        else None  # suppress when there's plenty of budget
    ),
)
```

## How It Works

`ToolBudgetToolset` extends Pydantic AI's [`WrapperToolset`](https://ai.pydantic.dev/toolsets/#wrapping-a-toolset). After each successful tool call, it wraps the result in a `ToolReturn` with a `.content` field containing the budget reminder. The framework converts this to a `UserPromptPart` placed after the tool result in the model request — exactly where the model reads it before deciding what to do next.

Counts reset automatically between agent runs.

## API

### `ToolBudgetToolset`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wrapped` | `AbstractToolset` | *required* | The toolset to wrap |
| `limits` | `dict[str, int]` | `{}` | Per-tool call limits |
| `default_limit` | `int \| None` | `None` | Default limit for unlisted tools |
| `threshold` | `int \| None` | `None` | Only remind when remaining <= threshold |
| `formatter` | `(name, used, limit) -> str \| None` | `None` | Custom reminder formatter |

## License

MIT
