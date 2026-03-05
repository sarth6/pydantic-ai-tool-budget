"""VCR-recorded integration tests using Gemini.

These tests record real LLM API calls and replay them from cassettes.
To re-record: GEMINI_API_KEY=... uv run pytest tests/test_vcr_gemini.py --record-mode=all
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_tool_budget import ToolBudgetToolset

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


# --- Helper ---


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


# --- Tests ---


@pytest.mark.vcr()
async def test_gemini_budget_reminders(allow_model_requests: None, gemini_api_key: str) -> None:
    """Basic integration: budget reminders appear in Gemini conversation."""
    toolset: FunctionToolset[None] = FunctionToolset([get_capital])
    budget_toolset: ToolBudgetToolset[None] = ToolBudgetToolset(
        wrapped=toolset,
        limits={"get_capital": 3},
    )

    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, toolsets=[budget_toolset])

    result = await agent.run("What is the capital of France? Use the get_capital tool.")

    messages = result.all_messages()
    reminders = _budget_parts(messages)

    # At least one budget reminder should be injected
    assert len(reminders) >= 1
    assert "get_capital" in reminders[0]
    assert "1/3" in reminders[0]


@pytest.mark.vcr()
async def test_gemini_multiple_tools(allow_model_requests: None, gemini_api_key: str) -> None:
    """Multiple tools each get their own budget reminder."""
    toolset: FunctionToolset[None] = FunctionToolset([get_capital, get_population])
    budget_toolset: ToolBudgetToolset[None] = ToolBudgetToolset(
        wrapped=toolset,
        limits={"get_capital": 5, "get_population": 3},
    )

    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, toolsets=[budget_toolset])

    result = await agent.run(
        "What is the capital of France, and what is its population? Use the get_capital and get_population tools."
    )

    messages = result.all_messages()
    reminders = _budget_parts(messages)

    # Should have reminders for both tools
    tool_names_in_reminders = [r.split(":")[0] for r in reminders]
    assert "get_capital" in tool_names_in_reminders
    assert "get_population" in tool_names_in_reminders


@pytest.mark.vcr()
async def test_gemini_threshold_filtering(allow_model_requests: None, gemini_api_key: str) -> None:
    """Threshold suppresses reminders when budget is comfortable."""
    toolset: FunctionToolset[None] = FunctionToolset([get_capital])
    budget_toolset: ToolBudgetToolset[None] = ToolBudgetToolset(
        wrapped=toolset,
        limits={"get_capital": 10},
        threshold=2,  # only remind when remaining <= 2
    )

    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, toolsets=[budget_toolset])

    result = await agent.run("What is the capital of France? Use the get_capital tool.")

    messages = result.all_messages()
    reminders = _budget_parts(messages)

    # With limit=10 and threshold=2, after 1 call (9 remaining > 2), no reminder
    assert len(reminders) == 0


@pytest.mark.vcr()
async def test_gemini_custom_formatter(allow_model_requests: None, gemini_api_key: str) -> None:
    """Custom formatter produces custom reminder text."""

    def urgent_formatter(name: str, used: int, limit: int) -> str | None:
        remaining = limit - used
        if remaining <= 2:
            return f"WARNING: Only {remaining} {name} calls left!"
        return f"{name} budget OK ({remaining}/{limit} remaining)"

    toolset: FunctionToolset[None] = FunctionToolset([get_capital])
    budget_toolset: ToolBudgetToolset[None] = ToolBudgetToolset(
        wrapped=toolset,
        limits={"get_capital": 5},
        formatter=urgent_formatter,
    )

    model = GoogleModel("gemini-2.0-flash", provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model, toolsets=[budget_toolset])

    result = await agent.run("What is the capital of France? Use the get_capital tool.")

    messages = result.all_messages()
    all_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    if "budget" in part.content.lower() or "remaining" in part.content.lower():
                        all_parts.append(part.content)

    assert len(all_parts) >= 1
    assert "budget OK" in all_parts[0] or "WARNING" in all_parts[0]
