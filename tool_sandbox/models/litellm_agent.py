# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""LiteLLM-based universal agent for ToolSandbox."""

import logging
from typing import Optional

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    from_litellm_response_to_messages,
    to_litellm_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.configs.models import APIModelConfig
from tool_sandbox.models.base_role import BaseRole
from tool_sandbox.models.litellm import LiteLLMModel


class LiteLLMAgent(BaseRole):
    """Universal agent using LiteLLM backend supporting 100+ model providers."""

    role_type: RoleType = RoleType.AGENT

    def __init__(self, model_name: Optional[str] = None, config: Optional[APIModelConfig] = None) -> None:
        """Initialize the LiteLLM agent.

        Args:
            model_name: LiteLLM model name (e.g., "gpt-4o", "claude-3-sonnet-20240229")
            config: Optional API model configuration. If not provided, creates default config.
        """
        self.logger = logging.getLogger(__name__)

        # Create default config if not provided
        if config is None:
            config = APIModelConfig()

        # Override model name if provided
        if model_name is not None:
            config.model_name = model_name

        self.model_name = config.model_name
        self.litellm_model = LiteLLMModel(config=config, name=f"LiteLLMAgent-{self.model_name}", logger=self.logger)

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message.

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 LiteLLM API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: list[Message] = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)

        # Keep only relevant messages
        messages = self.filter_messages(messages=messages)

        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return

        # Get tools if most recent message is from user or execution environment
        available_tools = self.get_available_tools()
        tools = None

        if (
            messages[-1].sender == RoleType.USER or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
        ) and available_tools:
            # Convert tools to OpenAI format (LiteLLM uses OpenAI-compatible format)
            tools = convert_to_openai_tools(available_tools)

        # Convert ToolSandbox messages to LiteLLM format
        litellm_messages = to_litellm_messages(messages)

        # Call model
        response = self.litellm_model.query(messages=litellm_messages, tools=tools)

        # Convert response back to ToolSandbox messages
        current_context = get_current_context()
        agent_to_execution_tool_name = current_context.get_agent_to_execution_facing_tool_name()
        available_tool_names = set(available_tools.keys())

        response_messages = from_litellm_response_to_messages(
            response=response,
            sender=self.role_type,
            available_tool_names=available_tool_names,
            agent_to_execution_facing_tool_name=agent_to_execution_tool_name,
        )

        # If no messages generated (shouldn't happen), create a default response
        if not response_messages:
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content="I apologize, but I couldn't process that request.",
                )
            ]

        self.add_messages(response_messages)


# Model-specific agent classes for easy configuration
class GPT4LiteLLMAgent(LiteLLMAgent):
    """GPT-4o agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize GPT-4o agent."""
        super().__init__(model_name="gpt-4o")


class GPT4oMiniLiteLLMAgent(LiteLLMAgent):
    """GPT-4o-mini agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize GPT-4o-mini agent."""
        super().__init__(model_name="gpt-4o-mini")


class ClaudeLiteLLMAgent(LiteLLMAgent):
    """Claude 3 Sonnet agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize Claude 3 Sonnet agent."""
        super().__init__(model_name="claude-3-sonnet-20240229")


class ClaudeHaikuLiteLLMAgent(LiteLLMAgent):
    """Claude 3 Haiku agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize Claude 3 Haiku agent."""
        super().__init__(model_name="claude-3-haiku-20240307")


class ClaudeOpusLiteLLMAgent(LiteLLMAgent):
    """Claude 3 Opus agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize Claude 3 Opus agent."""
        super().__init__(model_name="claude-3-opus-20240229")


class GeminiProLiteLLMAgent(LiteLLMAgent):
    """Gemini Pro agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize Gemini Pro agent."""
        super().__init__(model_name="gemini/gemini-pro")


class GeminiFlashLiteLLMAgent(LiteLLMAgent):
    """Gemini Flash agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize Gemini Flash agent."""
        super().__init__(model_name="gemini/gemini-1.5-flash")


class O3MiniLiteLLMAgent(LiteLLMAgent):
    """o3-mini agent using LiteLLM."""

    def __init__(self) -> None:
        """Initialize o3-mini agent."""
        super().__init__(model_name="o3-mini")
