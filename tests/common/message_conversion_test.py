# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for tool_sandbox.roles.message_conversion"""

from typing import Tuple

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    python_code_to_openai_tool_call,
)


@pytest.fixture
def python_code_and_openai_too_call() -> Tuple[str, ChatCompletionMessageToolCall]:
    """Contains test datapoint for OpenAI tool call and python code conversion

    Returns:
        A tuple of ChatCompletionMessageToolCall and corresponding python code
    """
    tool_id = "call_1293YXHD"
    name = "test"
    arguments = '{"a": 42}'
    return (
        "call_1293YXHD_parameters = {'a': 42}\n"
        "call_1293YXHD_response = test(**call_1293YXHD_parameters)\n"
        "print(repr(call_1293YXHD_response))"
    ), ChatCompletionMessageToolCall(
        id=tool_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_openai_tool_call_to_python_code(
    python_code_and_openai_too_call: Tuple[str, ChatCompletionMessageToolCall],
) -> None:
    python_code, tool_call = python_code_and_openai_too_call
    assert (
        openai_tool_call_to_python_code(
            tool_call, {"test"}, execution_facing_tool_name=None
        )
        == python_code
    )


def test_python_code_to_openai_tool_call(
    python_code_and_openai_too_call: Tuple[str, ChatCompletionMessageToolCall],
) -> None:
    python_code, tool_call = python_code_and_openai_too_call
    assert python_code_to_openai_tool_call(python_code, None) == tool_call


def test_message_construction() -> None:
    message = Message(sender="AGENT", recipient="USER", content="Hi")
    # Test sender recipient converter
    assert isinstance(message.sender, RoleType)
    assert isinstance(message.recipient, RoleType)
    # Test Message post init hook
    assert message.visible_to == [RoleType.AGENT, RoleType.USER]
    # Test visible_to converter
    message = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content="Hi",
        visible_to=[RoleType.USER],
    )
    assert message.visible_to == [RoleType.USER]
