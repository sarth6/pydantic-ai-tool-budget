# pydantic-ai-tool-budget

## How it works

`budgeted()` wraps tool functions with `@functools.wraps`, so `inspect.signature()` sees the original's parameters (pydantic-ai's `_function_schema.py:100` follows `__wrapped__`). The wrapper returns `ToolReturn(content=...)` which becomes a `UserPromptPart` after the tool result in `_agent_graph.py:1360-1407`.

Sync/async detection: `is_async_callable` checks the actual function, NOT `__wrapped__`. So `budgeted()` creates a sync wrapper for sync functions and an async wrapper for async functions to preserve `run_in_executor` behavior.

## Testing

- Unit tests use `TestModel()` (must be instantiated, not passed as class)
- VCR tests require the `allow_model_requests` fixture even in replay mode
- `GoogleModel` takes `provider=GoogleProvider(api_key=key)`, not `api_key=key` directly
- To re-record cassettes: `GEMINI_API_KEY=... uv run pytest tests/test_vcr_gemini.py --record-mode=all`

## Legacy API

`ToolBudgetToolset` (WrapperToolset approach) still exists for backwards compat and bulk configuration via `default_limit`. Prefer `budgeted()` for new code.
