"""WrapperToolset that injects per-tool budget reminders after tool results."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import ToolReturn
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.messages import UserContent
from pydantic_ai.toolsets import WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool


@dataclass
class ToolBudgetToolset(WrapperToolset[AgentDepsT]):
    """Wraps a toolset to inject per-tool budget reminders after tool results.

    After each successful tool call, a reminder is appended as a ``UserPromptPart``
    alongside the tool result, telling the model how many calls remain for that tool.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.toolsets import FunctionToolset
        from pydantic_ai_tool_budget import ToolBudgetToolset

        def search(query: str) -> str:
            return f"Results for {query}"

        agent = Agent(
            'openai:gpt-4o',
            toolsets=[
                ToolBudgetToolset(
                    wrapped=FunctionToolset([search]),
                    limits={'search': 5},
                ),
            ],
        )
    """

    limits: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    """Per-tool call limits. Keys are tool names, values are max allowed calls."""

    default_limit: int | None = None
    """Default limit applied to tools not listed in ``limits``. ``None`` means no limit."""

    threshold: int | None = None
    """Only inject a reminder when remaining calls <= ``threshold``.

    ``None`` (default) means always remind after every tool call that has a limit.
    """

    formatter: Callable[[str, int, int], str | None] | None = None
    """Custom formatter: ``(tool_name, used, limit) -> reminder text``.

    Return ``None`` to suppress the reminder for that call.
    """

    _counts: dict[str, int] = field(default_factory=lambda: dict[str, int](), init=False, repr=False)
    _run_id: str | None = field(default=None, init=False, repr=False)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        # Reset counts when a new run starts
        if ctx.run_id is not None and ctx.run_id != self._run_id:
            self._counts.clear()
            self._run_id = ctx.run_id

        # Execute the actual tool (ModelRetry propagates without counting)
        result = await super().call_tool(name, tool_args, ctx, tool)

        # Count this successful call
        self._counts[name] = self._counts.get(name, 0) + 1
        used = self._counts[name]

        # Determine the limit for this tool
        limit = self.limits.get(name, self.default_limit)
        if limit is None:
            return result

        remaining = max(limit - used, 0)

        # Apply threshold filter
        if self.threshold is not None and remaining > self.threshold:
            return result

        # Format the reminder
        if self.formatter is not None:
            text = self.formatter(name, used, limit)
            if text is None:
                return result
        else:
            text = f"{name}: {used}/{limit} calls used, {remaining} remaining."
            if remaining == 0:
                text += " This tool's budget is exhausted."

        # Attach reminder as ToolReturn.content → becomes UserPromptPart
        return _attach_content(result, text)


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
