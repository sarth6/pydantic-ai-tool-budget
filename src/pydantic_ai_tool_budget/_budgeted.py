"""Decorator that adds per-tool budget reminders to Pydantic AI tool functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

from pydantic_ai import ToolReturn
from pydantic_ai.messages import UserContent


def budgeted(
    func: Callable[..., Any],
    *,
    limit: int,
    threshold: int | None = None,
    formatter: Callable[[str, int, int], str | None] | None = None,
) -> Callable[..., Any]:
    """Wrap a tool function with per-call budget reminders.

    After each call, a ``ToolReturn`` with a ``.content`` reminder is returned,
    which pydantic-ai surfaces as a ``UserPromptPart`` the model sees alongside the tool result.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai_tool_budget import budgeted

        def search(query: str) -> str:
            return f"Results for {query}"

        agent = Agent('openai:gpt-4o', tools=[budgeted(search, limit=5)])

    Args:
        func: The tool function to wrap. Can be sync or async, with or without ``RunContext``.
        limit: Maximum number of calls before the budget is exhausted.
        threshold: Only inject a reminder when remaining calls <= ``threshold``.
            ``None`` (default) means always remind.
        formatter: Custom ``(tool_name, used, limit) -> reminder text``.
            Return ``None`` to suppress the reminder for that call.
    """
    name = func.__name__
    state: dict[str, int] = {"count": 0}

    def _make_reminder() -> str | None:
        used = state["count"]
        remaining = max(limit - used, 0)

        if threshold is not None and remaining > threshold:
            return None

        if formatter is not None:
            return formatter(name, used, limit)

        text = f"{name}: {used}/{limit} calls used, {remaining} remaining."
        if remaining == 0:
            text += " This tool's budget is exhausted."
        return text

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            state["count"] += 1
            text = _make_reminder()
            if text is None:
                return result
            return _attach_content(result, text)

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            state["count"] += 1
            text = _make_reminder()
            if text is None:
                return result
            return _attach_content(result, text)

        return sync_wrapper


def _attach_content(result: Any, text: str) -> ToolReturn:
    """Wrap or augment a tool result with budget reminder content."""
    if isinstance(result, ToolReturn):
        existing = result.content
        new_content: str | Sequence[UserContent]
        if existing is None:
            new_content = text
        elif isinstance(existing, str):
            new_content = f"{existing}\n{text}"
        else:
            new_content = [*existing, text]
        return ToolReturn(
            return_value=result.return_value,
            content=new_content,
            metadata=result.metadata,
        )
    return ToolReturn(return_value=result, content=text)
