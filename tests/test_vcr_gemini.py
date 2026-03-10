"""VCR-recorded integration tests using Gemini.

These tests record real LLM API calls and replay them from cassettes.
To re-record: GEMINI_API_KEY=... uv run pytest tests/test_vcr_gemini.py --record-mode=all
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai_tool_budget import budgeted

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]


# --- Tool functions ---


def get_capital(country: str) -> str:
    """Get the capital city of a country."""
    capitals = {
        "france": "Paris",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
        "japan": "Tokyo",
    }
    return capitals.get(country.lower(), f"Unknown capital for {country}")


def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {
        "paris": "2.1 million",
        "berlin": "3.7 million",
        "rome": "2.8 million",
        "madrid": "3.3 million",
        "tokyo": "14 million",
    }
    return populations.get(city.lower(), f"Unknown population for {city}")


# --- Helpers ---


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


def _assert_tool_round_trip(messages: list[ModelMessage], tool_name: str) -> None:
    """Assert the standard 4-message tool call pattern exists."""
    # Message 0: user prompt
    assert isinstance(messages[0], ModelRequest)
    assert any(isinstance(p, UserPromptPart) for p in messages[0].parts)

    # Message 1: model calls tool
    assert isinstance(messages[1], ModelResponse)
    tool_calls = [p for p in messages[1].parts if isinstance(p, ToolCallPart)]
    assert any(tc.tool_name == tool_name for tc in tool_calls)

    # Message 2: tool return + budget reminder
    assert isinstance(messages[2], ModelRequest)
    tool_returns = [p for p in messages[2].parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) >= 1

    # Final message: text response
    assert isinstance(messages[-1], ModelResponse)
    assert any(isinstance(p, TextPart) for p in messages[-1].parts)


# --- Tests ---


@pytest.mark.vcr()
async def test_gemini_budget_reminders(allow_model_requests: None, gemini_api_key: str) -> None:
    """Budget reminder appears as UserPromptPart after tool result in Gemini conversation."""
    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, tools=[budgeted(get_capital, limit=3)])

    result = await agent.run("What is the capital of France? Use the get_capital tool.")

    # Verify output mentions Paris
    assert "Paris" in result.output

    # Verify message structure
    messages = result.all_messages()
    _assert_tool_round_trip(messages, "get_capital")

    # Verify budget reminder is in the tool-return request
    tool_request = messages[2]
    assert isinstance(tool_request, ModelRequest)
    budget_parts = [p for p in tool_request.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)]
    assert len(budget_parts) == 1
    assert budget_parts[0].content == "get_capital: 1/3 calls used, 2 remaining."

    # Verify tool return has the actual result
    tool_returns = [p for p in tool_request.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == "Paris"


@pytest.mark.vcr()
async def test_gemini_multiple_tools(allow_model_requests: None, gemini_api_key: str) -> None:
    """Multiple tools each get their own independent budget reminder."""
    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(
        model,
        tools=[
            budgeted(get_capital, limit=5),
            budgeted(get_population, limit=3),
        ],
    )

    result = await agent.run(
        "What is the capital of France, and what is its population? Use the get_capital and get_population tools."
    )

    # Both tool budgets should appear
    reminders = _budget_parts(result.all_messages())
    tool_names_in_reminders = [r.split(":")[0] for r in reminders]
    assert "get_capital" in tool_names_in_reminders
    assert "get_population" in tool_names_in_reminders

    # Verify each reminder has correct format
    for reminder in reminders:
        assert "calls used" in reminder
        assert "remaining" in reminder


@pytest.mark.vcr()
async def test_gemini_threshold_filtering(allow_model_requests: None, gemini_api_key: str) -> None:
    """Threshold suppresses reminders when budget is comfortable."""
    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, tools=[budgeted(get_capital, limit=10, threshold=2)])

    result = await agent.run("What is the capital of France? Use the get_capital tool.")

    # With limit=10, threshold=2: after 1 call, 9 remaining > 2 → no reminder
    reminders = _budget_parts(result.all_messages())
    assert len(reminders) == 0

    # But the tool result itself should still be present
    messages = result.all_messages()
    tool_request = messages[2]
    assert isinstance(tool_request, ModelRequest)
    tool_returns = [p for p in tool_request.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == "Paris"

    # No UserPromptPart in the tool-return request (threshold suppressed it)
    user_parts = [p for p in tool_request.parts if isinstance(p, UserPromptPart)]
    assert len(user_parts) == 0


@pytest.mark.vcr()
async def test_gemini_custom_formatter(allow_model_requests: None, gemini_api_key: str) -> None:
    """Custom formatter controls reminder text."""

    def urgent_formatter(name: str, used: int, limit: int) -> str | None:
        remaining = limit - used
        if remaining <= 2:
            return f"WARNING: Only {remaining} {name} calls left!"
        return f"{name} budget OK ({remaining}/{limit} remaining)"

    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, tools=[budgeted(get_capital, limit=5, formatter=urgent_formatter)])

    result = await agent.run("What is the capital of France? Use the get_capital tool.")

    # After 1 call with limit=5: 4 remaining > 2, so we get "budget OK"
    messages = result.all_messages()
    tool_request = messages[2]
    assert isinstance(tool_request, ModelRequest)
    budget_parts = [p for p in tool_request.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)]
    assert len(budget_parts) == 1
    assert budget_parts[0].content == "get_capital budget OK (4/5 remaining)"
