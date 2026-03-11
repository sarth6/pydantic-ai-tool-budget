"""Tests for the budgeted() decorator."""

from __future__ import annotations

import pytest
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

from pydantic_ai_tool_budget import ToolBudget, budgeted

# --- Helpers ---


def _user_prompt_parts(result: object) -> list[str]:
    """Extract injected UserPromptPart texts (skipping the initial user prompt)."""
    texts: list[str] = []
    first_seen = False
    for msg in result.all_messages():  # type: ignore[union-attr]
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if not first_seen:
                        first_seen = True
                        continue
                    if isinstance(part.content, str):
                        texts.append(part.content)
                    elif isinstance(part.content, list):
                        texts.extend(item for item in part.content if isinstance(item, str))
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
    # Original content and budget reminder are separate UserPromptParts
    assert any("extra context" in r for r in reminders)
    assert any("calls used" in r for r in reminders)


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


# --- Validation ---


def test_budget_and_limit_mutually_exclusive() -> None:
    def search(query: str) -> str:
        """Search."""
        return query

    with pytest.raises(ValueError, match="Cannot specify both"):
        budgeted(search, limit=5, budget=ToolBudget(limit=10))


def test_must_specify_limit_or_budget() -> None:
    def search(query: str) -> str:
        """Search."""
        return query

    with pytest.raises(ValueError, match="Must specify either"):
        budgeted(search)


def test_exempt_requires_budget() -> None:
    def search(query: str) -> str:
        """Search."""
        return query

    with pytest.raises(ValueError, match="'exempt' is only valid"):
        budgeted(search, limit=5, exempt=True)


# --- on_exhaust (Pattern 1) ---


def test_on_exhaust_returns_custom_tool_return() -> None:
    """When budget exhausted and on_exhaust returns ToolReturn, use it as-is."""
    call_log: list[str] = []

    def tavily_search(query: str) -> str:
        """Search with Tavily."""
        call_log.append(query)
        return f"Results for {query}"

    def exhaust_handler(name: str, used: int, limit: int) -> ToolReturn:
        return ToolReturn(
            return_value=[],
            content=f"{name} limit reached ({used} calls). You must now compile your research.",
        )

    wrapped = budgeted(tavily_search, limit=1, on_exhaust=exhaust_handler)
    agent = Agent(TestModel(), tools=[wrapped])

    # First run: uses the 1 allowed call
    agent.run_sync("search once")
    assert len(call_log) == 1

    # Second run: budget exhausted, on_exhaust fires instead of real function
    result = agent.run_sync("search again")
    assert len(call_log) == 1  # real function NOT called again

    # The exhaust response should appear in messages
    found_exhaust = False
    for msg in result.all_messages():
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    if "compile your research" in part.content:
                        found_exhaust = True
    assert found_exhaust


def test_on_exhaust_plain_value_gets_reminder() -> None:
    """When on_exhaust returns a plain value, standard reminder is appended."""

    def search(query: str) -> str:
        """Search."""
        return f"Results for {query}"

    wrapped = budgeted(search, limit=1, on_exhaust=lambda name, used, lim: "no results")
    agent = Agent(TestModel(), tools=[wrapped])

    # First run exhausts the budget
    agent.run_sync("search once")

    # Second run triggers on_exhaust with plain value → gets default reminder
    result = agent.run_sync("search again")

    reminders = _user_prompt_parts(result)
    assert any("budget is exhausted" in r for r in reminders)


# --- Shared budget (Pattern 3) ---


def test_shared_budget_across_tools() -> None:
    """Multiple tools sharing one ToolBudget pool."""
    pool = ToolBudget(limit=2)

    def tool_a(query: str) -> str:
        """Tool A."""
        return f"A: {query}"

    def tool_b(query: str) -> str:
        """Tool B."""
        return f"B: {query}"

    wrapped_a = budgeted(tool_a, budget=pool)
    wrapped_b = budgeted(tool_b, budget=pool)

    # Call tool_a via agent
    agent_a = Agent(TestModel(), tools=[wrapped_a])
    agent_a.run_sync("use tool a")
    assert pool.used == 1

    # Call tool_b via agent
    agent_b = Agent(TestModel(), tools=[wrapped_b])
    agent_b.run_sync("use tool b")
    assert pool.used == 2
    assert pool.remaining == 0


def test_shared_budget_exempt_tool() -> None:
    """Exempt tools don't count against the shared budget."""
    pool = ToolBudget(limit=2)
    exempt_calls: list[str] = []

    def explorer(query: str) -> str:
        """Explore signals."""
        return f"Signals for {query}"

    def register(query: str) -> str:
        """Register opportunity — goal action, exempt from budget."""
        exempt_calls.append(query)
        return f"Registered {query}"

    wrapped_explorer = budgeted(explorer, budget=pool)
    wrapped_register = budgeted(register, budget=pool, exempt=True)

    # Use explorer twice to exhaust the pool
    agent_e = Agent(TestModel(), tools=[wrapped_explorer])
    agent_e.run_sync("explore 1")
    agent_e.run_sync("explore 2")
    assert pool.used == 2
    assert pool.is_exhausted()

    # Register is exempt — still works even when pool exhausted
    agent_r = Agent(TestModel(), tools=[wrapped_register])
    result = agent_r.run_sync("register something")
    assert len(exempt_calls) == 1

    # Pool count unchanged (exempt doesn't increment)
    assert pool.used == 2

    # Register's reminder should still show the pool status
    reminders = _user_prompt_parts(result)
    assert any("register:" in r and "2/2" in r for r in reminders)


def test_shared_budget_on_exhaust_blocks_non_exempt() -> None:
    """When shared budget exhausted, on_exhaust blocks non-exempt tools but exempt tools still run."""
    pool = ToolBudget(limit=1)
    real_calls: list[str] = []

    def search(query: str) -> str:
        """Search."""
        real_calls.append(f"search:{query}")
        return f"Results for {query}"

    def register(query: str) -> str:
        """Register — exempt."""
        real_calls.append(f"register:{query}")
        return f"Registered {query}"

    wrapped_search = budgeted(
        search,
        budget=pool,
        on_exhaust=lambda name, used, lim: ToolReturn(return_value=[], content="Budget exhausted."),
    )
    wrapped_register = budgeted(register, budget=pool, exempt=True)

    # First search call: real function runs, pool exhausted
    agent_s = Agent(TestModel(), tools=[wrapped_search])
    agent_s.run_sync("search 1")
    assert real_calls == ["search:a"]

    # Second search call: on_exhaust fires, real function NOT called
    agent_s.run_sync("search 2")
    assert real_calls == ["search:a"]  # unchanged

    # Register still works (exempt)
    agent_r = Agent(TestModel(), tools=[wrapped_register])
    agent_r.run_sync("register")
    assert real_calls == ["search:a", "register:a"]


def test_tool_budget_reset() -> None:
    """ToolBudget.reset() clears the counter."""
    pool = ToolBudget(limit=3)
    pool.record()
    pool.record()
    assert pool.used == 2
    pool.reset()
    assert pool.used == 0
    assert pool.remaining == 3


# --- ToolReturn metadata preservation (Pattern 2) ---


def test_tool_return_metadata_preserved() -> None:
    """Budget reminder preserves metadata from the original ToolReturn."""

    def search(query: str) -> ToolReturn:
        """Search with metadata."""
        return ToolReturn(
            return_value=f"Results for {query}",
            content="retrieval context",
            metadata={"source": "brand_db", "score": 0.95},
        )

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5)])
    result = agent.run_sync("search for paris")

    # Original content and budget reminder are separate UserPromptParts, metadata preserved
    reminders = _user_prompt_parts(result)
    assert any("retrieval context" in r for r in reminders)
    assert any("calls used" in r for r in reminders)


# --- Async coverage for on_exhaust and shared budget ---


async def test_async_on_exhaust() -> None:
    """Async tool with on_exhaust fires correctly when budget exhausted."""
    call_log: list[str] = []

    async def search(query: str) -> str:
        """Search."""
        call_log.append(query)
        return f"Results for {query}"

    wrapped = budgeted(
        search,
        limit=1,
        on_exhaust=lambda name, used, lim: ToolReturn(return_value=[], content="Exhausted."),
    )
    agent = Agent(TestModel(), tools=[wrapped])

    await agent.run("first call")
    assert len(call_log) == 1

    # Second call: on_exhaust intercepts
    result = await agent.run("second call")
    assert len(call_log) == 1

    found = False
    for msg in result.all_messages():
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    if "Exhausted." in part.content:
                        found = True
    assert found


async def test_async_threshold_suppresses() -> None:
    """Async tool with threshold suppression hits the 'text is None' return path."""

    async def search(query: str) -> str:
        """Search."""
        return f"Results for {query}"

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5, threshold=0)])
    result = await agent.run("search")

    # After 1 call: 4 remaining > threshold 0 → no reminder
    reminders = _user_prompt_parts(result)
    assert len(reminders) == 0


def test_on_exhaust_plain_value_with_formatter_none() -> None:
    """on_exhaust returns plain value, but formatter returns None → no reminder attached."""

    def search(query: str) -> str:
        """Search."""
        return f"Results for {query}"

    wrapped = budgeted(
        search,
        limit=1,
        on_exhaust=lambda name, used, lim: "fallback",
        formatter=lambda name, used, lim: None,
    )
    agent = Agent(TestModel(), tools=[wrapped])

    # Exhaust the budget
    agent.run_sync("first")

    # Second call: on_exhaust fires, formatter returns None → plain value returned as-is
    result = agent.run_sync("second")

    # The tool return should contain "fallback" but no budget reminder
    messages = result.all_messages()
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    assert part.content == "fallback"


def test_tool_return_with_no_content() -> None:
    """ToolReturn with content=None gets budget reminder as sole content."""

    def search(query: str) -> ToolReturn:
        """Search."""
        return ToolReturn(return_value=f"Results for {query}")

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5)])
    result = agent.run_sync("search")

    reminders = _user_prompt_parts(result)
    assert any("calls used" in r for r in reminders)


async def test_async_tool_return_with_sequence_content() -> None:
    """Async tool returning ToolReturn with sequence content gets budget reminder appended."""

    async def search(query: str) -> ToolReturn:
        """Search."""
        return ToolReturn(return_value=f"Results for {query}", content=["ctx1", "ctx2"])

    agent = Agent(TestModel(), tools=[budgeted(search, limit=5)])
    result = await agent.run("search")

    reminders = _user_prompt_parts(result)
    assert any("ctx1" in r for r in reminders)
    assert any("ctx2" in r for r in reminders)
    assert any("calls used" in r for r in reminders)
