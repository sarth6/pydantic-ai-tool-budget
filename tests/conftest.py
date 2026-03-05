"""Test configuration for pydantic-ai-tool-budget."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pydantic_ai.models
import pytest

# Block real model requests by default — VCR cassettes replay them
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or "mock-api-key"


@pytest.fixture()
def allow_model_requests() -> Iterator[None]:
    """Enable real model requests for VCR recording."""
    pydantic_ai.models.ALLOW_MODEL_REQUESTS = True
    yield
    pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


def pytest_recording_configure(config: Any, vcr: Any) -> None:
    from . import json_body_serializer

    vcr.register_serializer("yaml", json_body_serializer)


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "ignore_localhost": True,
        "filter_headers": ["authorization", "x-api-key", "x-goog-api-key"],
        "decode_compressed_response": True,
    }
