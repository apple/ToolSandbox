# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for the Cohere agent."""

import pytest

from tool_sandbox.common.tool_discovery import ToolBackend, get_all_tools
from tool_sandbox.roles.cohere_api_agent import to_cohere_tool


@pytest.mark.parametrize("tool_backend", ToolBackend)
def test_tool_conversion(tool_backend: ToolBackend) -> None:
    """Ensure that all our tools can be converted to the Cohere format."""
    name_to_tool = get_all_tools(preferred_tool_backend=tool_backend)
    for name, tool in name_to_tool.items():
        assert to_cohere_tool(name=name, tool=tool) is not None
