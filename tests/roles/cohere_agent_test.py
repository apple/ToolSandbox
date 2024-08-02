# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for the Cohere agent."""

from typing import cast

import pytest
from openai.types.chat import ChatCompletionToolParam

from tool_sandbox.common.tool_conversion import convert_to_openai_tool
from tool_sandbox.common.tool_discovery import ToolBackend, get_all_tools
from tool_sandbox.roles.cohere_agent import to_cohere_tool


@pytest.mark.parametrize("tool_backend", ToolBackend)
def test_tool_conversion(tool_backend: ToolBackend) -> None:
    """Ensure that all our tools can be converted to the Cohere format."""
    name_to_tool = get_all_tools(preferred_tool_backend=tool_backend)
    for tool in name_to_tool.values():
        openai_tool = convert_to_openai_tool(tool)
        assert to_cohere_tool(cast(ChatCompletionToolParam, openai_tool)) is not None
