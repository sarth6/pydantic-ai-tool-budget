"""Decorator that adds per-tool budget reminders to Pydantic AI tool functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

from pydantic_ai import ToolReturn
from pydantic_ai.messages import UserContent

from pydantic_ai_tool_budget._budget import ToolBudget


def budgeted(
    func: Callable[..., Any],
    *,
    limit: int | None = None,
    budget: ToolBudget | None = None,
    exempt: bool = False,
    threshold: int | None = None,
    formatter: Callable[[str, int, int], str | None] | None = None,
    on_exhaust: Callable[[str, int, int], Any] | None = None,
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
            Mutually exclusive with ``budget``.
        budget: A shared :class:`ToolBudget` instance. When provided, this tool draws from
            the shared pool instead of tracking its own count. Mutually exclusive with ``limit``.
        exempt: If ``True`` and ``budget`` is provided, this tool does **not** count against
            the shared budget but still shows remaining-call reminders. Only valid with ``budget``.
        threshold: Only inject a reminder when remaining calls <= ``threshold``.
            ``None`` (default) means always remind.
        formatter: Custom ``(tool_name, used, limit) -> reminder text``.
            Return ``None`` to suppress the reminder for that call.
        on_exhaust: Called instead of the real tool function when the budget is exhausted.
            Receives ``(tool_name, used, limit)`` and returns the value to use as the tool result.
            If it returns a :class:`ToolReturn`, that is used as-is (no budget reminder appended).
            If it returns a plain value, the standard budget reminder is appended.
    """
    if limit is not None and budget is not None:
        raise ValueError("Cannot specify both 'limit' and 'budget'")
    if limit is None and budget is None:
        raise ValueError("Must specify either 'limit' or 'budget'")
    if exempt and budget is None:
        raise ValueError("'exempt' is only valid when 'budget' is provided")

    name = func.__name__
    state: dict[str, int] = {"count": 0}

    def _effective_limit() -> int:
        if budget is not None:
            return budget.limit
        assert limit is not None
        return limit

    def _current_used() -> int:
        if budget is not None:
            return budget.used
        return state["count"]

    def _is_exhausted() -> bool:
        if budget is not None:
            return budget.is_exhausted()
        return state["count"] >= _effective_limit()

    def _increment() -> None:
        if budget is not None:
            if not exempt:
                budget.record()
        else:
            state["count"] += 1

    def _make_reminder() -> str | None:
        used = _current_used()
        lim = _effective_limit()
        remaining = max(lim - used, 0)

        if threshold is not None and remaining > threshold:
            return None

        if formatter is not None:
            return formatter(name, used, lim)

        text = f"{name}: {used}/{lim} calls used, {remaining} remaining."
        if remaining == 0:
            text += " This tool's budget is exhausted."
        return text

    def _handle_exhaust() -> Any | None:
        """If exhausted and on_exhaust is set and tool is not exempt, return exhaust result."""
        if not _is_exhausted() or on_exhaust is None or exempt:
            return None
        lim = _effective_limit()
        used = _current_used()
        exhaust_result = on_exhaust(name, used, lim)
        if isinstance(exhaust_result, ToolReturn):
            return exhaust_result
        # Wrap plain value with standard reminder
        text = _make_reminder()
        if text is None:
            return exhaust_result
        return _attach_content(exhaust_result, text)

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            exhaust = _handle_exhaust()
            if exhaust is not None:
                return exhaust
            result = await func(*args, **kwargs)
            _increment()
            text = _make_reminder()
            if text is None:
                return result
            return _attach_content(result, text)

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            exhaust = _handle_exhaust()
            if exhaust is not None:
                return exhaust
            result = func(*args, **kwargs)
            _increment()
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
            new_content = [existing, text]
        else:
            new_content = [*existing, text]
        return ToolReturn(
            return_value=result.return_value,
            content=new_content,
            metadata=result.metadata,
        )
    return ToolReturn(return_value=result, content=text)
