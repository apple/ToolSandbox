# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for LiteLLM agent."""

import pytest

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.configs.models import APIModelConfig
from tool_sandbox.models.litellm_agent import (
    ClaudeLiteLLMAgent,
    GPT4LiteLLMAgent,
    GeminiProLiteLLMAgent,
    LiteLLMAgent,
    O3MiniLiteLLMAgent,
)


def test_litellm_agent_initialization() -> None:
    """Test LiteLLM agent initialization with different configurations."""
    # Test with model name
    agent = LiteLLMAgent(model_name="gpt-3.5-turbo")
    assert agent.model_name == "gpt-3.5-turbo"
    assert agent.role_type == RoleType.AGENT

    # Test with custom config
    config = APIModelConfig(model_name="claude-3-haiku-20240307", temperature=0.5)
    agent = LiteLLMAgent(config=config)
    assert agent.model_name == "claude-3-haiku-20240307"

    # Test model name override
    agent = LiteLLMAgent(model_name="gpt-4o", config=config)
    assert agent.model_name == "gpt-4o"  # Should override config

    # Test default initialization
    agent = LiteLLMAgent()
    assert agent.model_name == "gpt4o"  # Default from APIModelConfig


def test_model_specific_agents() -> None:
    """Test that model-specific agent classes have correct model names."""
    test_cases = [
        (GPT4LiteLLMAgent, "gpt-4o"),
        (ClaudeLiteLLMAgent, "claude-3-sonnet-20240229"),
        (GeminiProLiteLLMAgent, "gemini/gemini-pro"),
        (O3MiniLiteLLMAgent, "o3-mini"),
    ]

    for agent_class, expected_model in test_cases:
        agent = agent_class()
        assert agent.model_name == expected_model
        assert agent.role_type == RoleType.AGENT
        assert isinstance(agent, LiteLLMAgent)


def test_litellm_agent_has_required_methods() -> None:
    """Test that LiteLLM agent has all required methods from BaseRole."""
    agent = LiteLLMAgent(model_name="gpt-3.5-turbo")

    # Check that it has the required methods
    assert hasattr(agent, "respond")
    assert hasattr(agent, "get_messages")
    assert hasattr(agent, "add_messages")
    assert hasattr(agent, "messages_validation")
    assert hasattr(agent, "filter_messages")
    assert hasattr(agent, "get_available_tools")

    # Check role type
    assert agent.role_type == RoleType.AGENT
