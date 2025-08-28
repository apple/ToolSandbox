# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for LiteLLM model enhancements."""

from unittest.mock import Mock, patch

import pytest

from tool_sandbox.configs.models import APIModelConfig
from tool_sandbox.models.litellm import LiteLLMModel


def test_litellm_model_initialization() -> None:
    """Test LiteLLM model initialization."""
    config = APIModelConfig(model_name="gpt-3.5-turbo")
    logger = Mock()

    model = LiteLLMModel(config=config, name="test-model", logger=logger)

    assert model.name == "test-model"
    assert model.config == config
    assert model.model_name == "gpt-3.5-turbo"
    assert model.logger == logger


def test_query_with_tools_method_exists() -> None:
    """Test that query_with_tools method exists and has correct signature."""
    config = APIModelConfig(model_name="gpt-3.5-turbo")
    logger = Mock()

    model = LiteLLMModel(config=config, name="test-model", logger=logger)

    # Check method exists
    assert hasattr(model, "query_with_tools")
    assert callable(model.query_with_tools)


@patch('tool_sandbox.models.litellm.litellm.completion')
@patch('tool_sandbox.models.litellm.litellm.utils.token_counter')
@patch('tool_sandbox.models.litellm.litellm.cost_calculator.completion_cost')
def test_query_with_tools_basic_functionality(mock_cost: Mock, mock_token_counter: Mock, mock_completion: Mock) -> None:
    """Test basic functionality of query_with_tools method."""
    # Setup mocks
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_completion.return_value = mock_response
    mock_token_counter.return_value = 10
    mock_cost.return_value = 0.01

    config = APIModelConfig(model_name="gpt-3.5-turbo")
    logger = Mock()
    model = LiteLLMModel(config=config, name="test-model", logger=logger)

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    # Call the method
    response = model.query_with_tools(messages=messages, tools=tools)

    # Verify the response
    assert response == mock_response

    # Verify litellm.completion was called with correct arguments
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args[1]["model"] == "gpt-3.5-turbo"
    assert call_args[1]["messages"] == messages
    assert call_args[1]["tools"] == tools
    assert call_args[1]["stream"] is False


@patch('tool_sandbox.models.litellm.litellm.completion')
@patch('tool_sandbox.models.litellm.litellm.utils.token_counter')
@patch('tool_sandbox.models.litellm.litellm.cost_calculator.completion_cost')
def test_query_with_tools_without_tools(mock_cost: Mock, mock_token_counter: Mock, mock_completion: Mock) -> None:
    """Test query_with_tools method without tools."""
    # Setup mocks
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_completion.return_value = mock_response
    mock_token_counter.return_value = 10
    mock_cost.return_value = 0.01

    config = APIModelConfig(model_name="gpt-3.5-turbo")
    logger = Mock()
    model = LiteLLMModel(config=config, name="test-model", logger=logger)

    messages = [{"role": "user", "content": "Hello"}]

    # Call the method without tools
    response = model.query_with_tools(messages=messages, tools=None)

    # Verify the response
    assert response == mock_response

    # Verify tools was not passed to litellm.completion
    call_args = mock_completion.call_args
    assert "tools" not in call_args[1] or call_args[1]["tools"] is None
