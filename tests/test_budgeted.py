"""Tests for the budgeted() decorator."""

from __future__ import annotations

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from pydantic_ai_tool_budget import budgeted

# --- Helpers ---


def _user_prompt_parts(result: object) -> list[str]:
    """Extract injected UserPromptPart texts (skipping the initial user prompt)."""
    texts: list[str] = []
    first_seen = False
    for msg in result.all_messages():  # type: ignore[union-attr]
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    if not first_seen:
                        first_seen = True
                        continue
                    texts.append(part.content)
    return texts


# --- Core behavior ---


def test_basic_reminder() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=3)])
    result = agent.run_sync("search for paris")

    assert result.output == '{"search":"Results for a"}'

    # Verify message structure: request → response(tool_call) → request(tool_return + budget) → response(text)
    messages = result.all_messages()
    assert len(messages) == 4

    # First message: user prompt
    assert isinstance(messages[0], ModelRequest)
    assert len(messages[0].parts) == 1
    assert isinstance(messages[0].parts[0], UserPromptPart)

    # Second message: model calls the tool
    assert isinstance(messages[1], ModelResponse)
    assert any(isinstance(p, ToolCallPart) for p in messages[1].parts)

    # Third message: tool result + budget reminder
    assert isinstance(messages[2], ModelRequest)
    parts = messages[2].parts
    tool_returns = [p for p in parts if isinstance(p, ToolReturnPart)]
    budget_parts = [p for p in parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == "Results for a"
    assert len(budget_parts) == 1
    assert budget_parts[0].content == "search: 1/3 calls used, 2 remaining."

    # Fourth message: model's final text response
    assert isinstance(messages[3], ModelResponse)
    assert any(isinstance(p, TextPart) for p in messages[3].parts)


def test_exhausted_message() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=1)])
    result = agent.run_sync("search for paris")

    reminders = _user_prompt_parts(result)
    assert len(reminders) >= 1
    assert "search: 1/1 calls used, 0 remaining." in reminders[0]
    assert "This tool's budget is exhausted." in reminders[0]


# --- Threshold ---


def test_threshold_suppresses_when_comfortable() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5, threshold=2)])
    result = agent.run_sync("search for paris")

    # After 1 call: 4 remaining > threshold 2 → no reminder injected
    reminders = _user_prompt_parts(result)
    assert len(reminders) == 0


def test_threshold_shows_when_tight() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=2, threshold=2)])
    result = agent.run_sync("search for paris")

    # After 1 call: 1 remaining <= threshold 2 → reminder shown
    reminders = _user_prompt_parts(result)
    assert len(reminders) >= 1
    assert "search: 1/2 calls used, 1 remaining." in reminders[0]


# --- Custom formatter ---


def test_custom_formatter() -> None:
    def my_fmt(name: str, used: int, limit: int) -> str | None:
        return f"CUSTOM: {name} has {limit - used} left"

    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5, formatter=my_fmt)])
    result = agent.run_sync("search for paris")

    reminders = _user_prompt_parts(result)
    assert any("CUSTOM: search has 4 left" in r for r in reminders)


def test_formatter_returning_none_suppresses() -> None:
    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(
        TestModel(),
        tools=[budgeted(search, limit=5, formatter=lambda name, used, lim: None)],
    )
    result = agent.run_sync("search for paris")

    reminders = _user_prompt_parts(result)
    assert len(reminders) == 0


# --- Edge cases ---


def test_undecorated_tool_no_reminders() -> None:
    """Sanity check: tools without budgeted() produce no reminders."""

    def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[search])
    result = agent.run_sync("search for paris")

    reminders = _user_prompt_parts(result)
    assert len(reminders) == 0


def test_existing_tool_return_content_merged() -> None:
    """When tool already returns ToolReturn with content, budget reminder is appended."""

    def search(query: str) -> ToolReturn:
        """Search the web."""
        return ToolReturn(return_value=f"Results for {query}", content="extra context")

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5)])
    result = agent.run_sync("search for paris")

    reminders = _user_prompt_parts(result)
    # Original content + budget reminder merged into one string
    assert any("extra context" in r and "calls used" in r for r in reminders)


# --- Async tools ---


async def test_async_tool() -> None:
    async def search(query: str) -> str:
        """Search the web."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=3)])
    result = await agent.run("search for paris")

    reminders = _user_prompt_parts(result)
    assert len(reminders) >= 1
    assert "search: 1/3 calls used, 2 remaining." in reminders[0]


# --- Wrapper transparency ---


def test_preserves_function_name() -> None:
    def my_special_tool(query: str) -> str:
        """A tool with a specific name."""
        return f"Results for {query}"

    wrapped = budgeted(my_special_tool, limit=5)
    assert wrapped.__name__ == "my_special_tool"
    assert wrapped.__doc__ == "A tool with a specific name."
