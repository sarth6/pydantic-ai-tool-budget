"""Tests for the budgeted() decorator."""

from __future__ import annotations

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_tool_budget import budgeted


def _budget_parts(messages: list[ModelMessage]) -> list[str]:
    """Extract budget reminder text from message history."""
    texts: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    if "calls used" in part.content or "budget" in part.content.lower():
                        texts.append(part.content)
    return texts


def _all_user_parts(messages: list[ModelMessage]) -> list[str]:
    """Extract ALL injected UserPromptParts (skip the initial prompt)."""
    texts: list[str] = []
    first_seen = False
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    if not first_seen:
                        first_seen = True
                        continue
                    texts.append(part.content)
    return texts


# --- Basic functionality ---


def test_basic_reminder() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=3)])
    result = agent.run_sync("search for paris")
    reminders = _budget_parts(result.all_messages())

    assert len(reminders) >= 1
    assert "search: 1/3 calls used, 2 remaining" in reminders[0]


def test_exhausted_message() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=1)])
    result = agent.run_sync("search for paris")
    reminders = _budget_parts(result.all_messages())

    assert len(reminders) >= 1
    assert "exhausted" in reminders[0].lower()


# --- Threshold ---


def test_threshold_suppresses_early() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5, threshold=2)])
    result = agent.run_sync("search for paris")
    reminders = _budget_parts(result.all_messages())

    # After 1 call: 4 remaining > threshold 2, no reminder
    assert len(reminders) == 0


def test_threshold_shows_when_tight() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=2, threshold=2)])
    result = agent.run_sync("search for paris")
    reminders = _budget_parts(result.all_messages())

    # After 1 call: 1 remaining <= threshold 2, reminder shown
    assert len(reminders) >= 1
    assert "search: 1/2 calls used, 1 remaining" in reminders[0]


# --- Custom formatter ---


def test_custom_formatter() -> None:
    def my_fmt(name: str, used: int, limit: int) -> str | None:
        return f"CUSTOM: {name} has {limit - used} left"

    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5, formatter=my_fmt)])
    result = agent.run_sync("search for paris")
    parts = _all_user_parts(result.all_messages())

    assert any("CUSTOM: search has 4 left" in p for p in parts)


def test_formatter_none_suppresses() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(
        TestModel(),
        tools=[budgeted(search, limit=5, formatter=lambda n, u, lim: None)],
    )
    result = agent.run_sync("search for paris")
    reminders = _budget_parts(result.all_messages())
    assert len(reminders) == 0


# --- No limit = no reminder ---


def test_no_budget_no_decorator() -> None:
    """Undecorated tools produce no reminders (sanity check)."""

    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[search])
    result = agent.run_sync("search for paris")
    reminders = _budget_parts(result.all_messages())
    assert len(reminders) == 0


# --- ToolReturn passthrough ---


def test_existing_tool_return_content() -> None:
    def search(query: str) -> ToolReturn:
        """Search the web."""
        return ToolReturn(return_value=f"Results for {query}", content="extra context")

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5)])
    result = agent.run_sync("search for paris")
    parts = _all_user_parts(result.all_messages())

    # Should have a part with both the original content and the budget reminder
    assert any("extra context" in p and "calls used" in p for p in parts)


# --- Async tools ---


async def test_async_tool() -> None:
    async def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=3)])
    result = await agent.run("search for paris")
    reminders = _budget_parts(result.all_messages())

    assert len(reminders) >= 1
    assert "search: 1/3 calls used, 2 remaining" in reminders[0]


# --- Type safety: function reference = no typo possible ---


def test_typo_is_name_error() -> None:
    """The whole point: budgeted(serach, limit=5) is a NameError, not a silent miss."""
    # This test just documents the design — if 'serach' were used,
    # Python raises NameError before any code runs.

    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    # Correct usage — compiles fine
    budgeted(search, limit=5)
    # budgeted(serach, limit=5)  # would be NameError
