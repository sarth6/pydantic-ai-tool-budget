"""Shared budget counter for use across multiple tools."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolBudget:
    """A shared call budget that can be referenced by multiple ``budgeted()`` wrappers.

    Use this when several tools should draw from the same pool of allowed calls.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai_tool_budget import ToolBudget, budgeted

        pool = ToolBudget(limit=20)

        agent = Agent(
            'openai:gpt-4o',
            tools=[
                budgeted(search_signals, budget=pool),
                budgeted(get_signal_details, budget=pool),
                # exempt: doesn't count against the pool, but still shows remaining
                budgeted(register_opportunity, budget=pool, exempt=True),
            ],
        )
    """

    limit: int
    """Maximum total calls allowed across all tools sharing this budget."""

    _count: int = field(default=0, init=False, repr=False)

    @property
    def used(self) -> int:
        """Number of calls made so far."""
        return self._count

    @property
    def remaining(self) -> int:
        """Calls remaining before exhaustion."""
        return max(self.limit - self._count, 0)

    def is_exhausted(self) -> bool:
        """Whether the budget has been fully consumed."""
        return self._count >= self.limit

    def record(self) -> None:
        """Record one call against the budget."""
        self._count += 1

    def reset(self) -> None:
        """Reset the counter to zero."""
        self._count = 0
