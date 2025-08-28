# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for LiteLLM message conversion functions."""

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message, to_litellm_messages


def test_to_litellm_messages_basic() -> None:
    """Test basic message conversion to LiteLLM format."""
    messages = [
        Message(sender=RoleType.SYSTEM, recipient=RoleType.AGENT, content="You are a helpful assistant."),
        Message(sender=RoleType.USER, recipient=RoleType.AGENT, content="Hello, how are you?"),
        Message(sender=RoleType.AGENT, recipient=RoleType.USER, content="I'm doing well, thank you!"),
    ]

    litellm_messages = to_litellm_messages(messages)

    assert len(litellm_messages) == 3

    # Check system message
    assert litellm_messages[0]["role"] == "system"
    assert litellm_messages[0]["content"] == "You are a helpful assistant."

    # Check user message
    assert litellm_messages[1]["role"] == "user"
    assert litellm_messages[1]["content"] == "Hello, how are you?"

    # Check assistant message
    assert litellm_messages[2]["role"] == "assistant"
    assert litellm_messages[2]["content"] == "I'm doing well, thank you!"


def test_to_litellm_messages_with_tool_calls() -> None:
    """Test message conversion with tool calls."""
    messages = [
        Message(sender=RoleType.USER, recipient=RoleType.AGENT, content="What's the weather?"),
        Message(
            sender=RoleType.AGENT,
            recipient=RoleType.EXECUTION_ENVIRONMENT,
            content="call_123_parameters = {'location': 'San Francisco'}\ncall_123_response = get_weather(**call_123_parameters)\nprint(repr(call_123_response))",
            openai_tool_call_id="call_123",
            openai_function_name="get_weather"
        ),
        Message(
            sender=RoleType.EXECUTION_ENVIRONMENT,
            recipient=RoleType.AGENT,
            content="Sunny, 72°F",
            openai_tool_call_id="call_123"
        ),
    ]

    litellm_messages = to_litellm_messages(messages)

    assert len(litellm_messages) == 3

    # Check user message
    assert litellm_messages[0]["role"] == "user"
    assert litellm_messages[0]["content"] == "What's the weather?"

    # Check tool call message
    assert litellm_messages[1]["role"] == "assistant"
    assert litellm_messages[1]["content"] is None
    assert "tool_calls" in litellm_messages[1]
    assert len(litellm_messages[1]["tool_calls"]) == 1

    tool_call = litellm_messages[1]["tool_calls"][0]
    assert tool_call["id"] == "call_123"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    # Check tool result message
    assert litellm_messages[2]["role"] == "tool"
    assert litellm_messages[2]["content"] == "Sunny, 72°F"
    assert litellm_messages[2]["tool_call_id"] == "call_123"


def test_to_litellm_messages_filters_irrelevant_messages() -> None:
    """Test that irrelevant message types are filtered out."""
    messages = [
        Message(sender=RoleType.USER, recipient=RoleType.AGENT, content="Hello"),
        Message(sender=RoleType.USER, recipient=RoleType.USER, content="This should be filtered"),
        Message(sender=RoleType.SYSTEM, recipient=RoleType.USER, content="This should also be filtered"),
        Message(sender=RoleType.AGENT, recipient=RoleType.USER, content="Hi there!"),
    ]

    litellm_messages = to_litellm_messages(messages)

    # Should only include the user->agent and agent->user messages
    assert len(litellm_messages) == 2
    assert litellm_messages[0]["role"] == "user"
    assert litellm_messages[0]["content"] == "Hello"
    assert litellm_messages[1]["role"] == "assistant"
    assert litellm_messages[1]["content"] == "Hi there!"
