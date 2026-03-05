"""Tests for ToolBudgetToolset."""

from __future__ import annotations

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_tool_budget import ToolBudgetToolset


def _search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"


def _search_with_content(query: str) -> ToolReturn:
    """Search the web, returning ToolReturn with existing content."""
    return ToolReturn(return_value=f"Results for {query}", content="extra context")


def _make_agent(
    *,
    limits: dict[str, int] | None = None,
    default_limit: int | None = None,
    threshold: int | None = None,
    formatter: object = None,
    use_tool_return: bool = False,
) -> Agent[None, str]:
    """Create a test agent with a search tool wrapped by ToolBudgetToolset."""
    tool_func = _search_with_content if use_tool_return else _search
    toolset: FunctionToolset[None] = FunctionToolset([tool_func])

    budget_toolset: ToolBudgetToolset[None] = ToolBudgetToolset(
        wrapped=toolset,
        limits=limits or {},
        default_limit=default_limit,
        threshold=threshold,
        formatter=formatter,  # type: ignore[arg-type]
    )

    return Agent(TestModel(), toolsets=[budget_toolset])


def _budget_parts(messages: list[ModelMessage]) -> list[UserPromptPart]:
    """Extract UserPromptParts containing budget reminder text."""
    parts: list[UserPromptPart] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and "calls used" in part.content:
                    parts.append(part)
    return parts


def _all_injected_parts(messages: list[ModelMessage]) -> list[UserPromptPart]:
    """Extract ALL UserPromptParts except the initial user prompt."""
    parts: list[UserPromptPart] = []
    first_seen = False
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if not first_seen:
                        first_seen = True
                        continue
                    parts.append(part)
    return parts


def test_basic_reminder() -> None:
    agent = _make_agent(limits={"_search": 3})
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    assert len(parts) >= 1
    content = parts[0].content
    assert isinstance(content, str)
    assert "_search: 1/3 calls used, 2 remaining" in content


def test_threshold_suppresses_early() -> None:
    agent = _make_agent(limits={"_search": 5}, threshold=2)
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    # After 1 call: 4 remaining > threshold 2, so no reminder
    assert len(parts) == 0


def test_threshold_shows_when_tight() -> None:
    agent = _make_agent(limits={"_search": 2}, threshold=2)
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    # After 1 call: 1 remaining <= threshold 2, so reminder shown
    assert len(parts) >= 1
    content = parts[0].content
    assert isinstance(content, str)
    assert "_search: 1/2 calls used, 1 remaining" in content


def test_custom_formatter() -> None:
    def my_formatter(name: str, used: int, limit: int) -> str | None:
        return f"CUSTOM: {name} has {limit - used} left"

    agent = _make_agent(limits={"_search": 5}, formatter=my_formatter)
    result = agent.run_sync("search for paris")
    parts = _all_injected_parts(result.all_messages())
    assert any(isinstance(p.content, str) and "CUSTOM: _search has 4 left" in p.content for p in parts)


def test_formatter_none_suppresses() -> None:
    def suppress_all(name: str, used: int, limit: int) -> str | None:
        return None

    agent = _make_agent(limits={"_search": 5}, formatter=suppress_all)
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    assert len(parts) == 0


def test_default_limit() -> None:
    agent = _make_agent(default_limit=3)
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    assert len(parts) >= 1
    content = parts[0].content
    assert isinstance(content, str)
    assert "_search: 1/3 calls used, 2 remaining" in content


def test_no_limit_no_reminder() -> None:
    agent = _make_agent()
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    assert len(parts) == 0


def test_exhausted_message() -> None:
    agent = _make_agent(limits={"_search": 1})
    result = agent.run_sync("search for paris")
    parts = _budget_parts(result.all_messages())
    assert len(parts) >= 1
    content = parts[0].content
    assert isinstance(content, str)
    assert "exhausted" in content.lower()


def test_counts_reset_per_run() -> None:
    agent = _make_agent(limits={"_search": 3})
    result1 = agent.run_sync("search for paris")
    parts1 = _budget_parts(result1.all_messages())
    assert len(parts1) >= 1
    content1 = parts1[0].content
    assert isinstance(content1, str)
    assert "1/3" in content1

    # Second run — counts should reset
    result2 = agent.run_sync("search for berlin")
    parts2 = _budget_parts(result2.all_messages())
    assert len(parts2) >= 1
    content2 = parts2[0].content
    assert isinstance(content2, str)
    assert "1/3" in content2


def test_existing_tool_return_content() -> None:
    agent = _make_agent(limits={"_search_with_content": 5}, use_tool_return=True)
    result = agent.run_sync("search for paris")
    parts = _all_injected_parts(result.all_messages())
    # Should have a part with both the original content and the budget reminder
    found = False
    for p in parts:
        content = p.content
        if isinstance(content, str) and "extra context" in content and "calls used" in content:
            found = True
            break
    assert found, f"Expected combined content, got: {[p.content for p in parts]}"
