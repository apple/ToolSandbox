# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for any model that conforms to OpenAI tool use API"""

import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

from nexusflowai import NexusflowAI
from nexusflowai.types.chat_completion import NexusflowAIChatCompletion
from nexusflowai.types.chat_completion_message import NexusflowAIChatCompletionMessage
from nexusflowai.types.chat_completion_message_tool_call import (
    NexusflowAIChatCompletionMessageToolCall,
)
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


class AtheneAPIAgent(BaseRole):
    """Agent role for an Athene-Custom agent, or any agent by NexusflowAI that is optimized for function calling."""

    role_type: RoleType = RoleType.AGENT
    model_name: str
    base_url: str
    api_key: str | None
    api_model_name: str

    message_history : Dict[str, Any] = {}

    def __init__(self) -> None:
        # We set the `base_url` explicitly here to avoid picking up the
        # `OPENAI_BASE_URL` environment variable that may be set for serving models as
        # OpenAI API compatible servers.
        self.nf_client: NexusflowAI = NexusflowAI(base_url=self.base_url, api_key=self.api_key)

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages.
        current_context = get_current_context()
        openai_messages, _ = to_openai_messages(messages)
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if not openai_response_message.tool_calls:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content
                )
            ]
        elif "chat" in openai_response_message.tool_calls[0].function.name:
            import json
            msg = json.loads(openai_response_message.tool_calls[0].function.arguments)["message"]
            self.message_history[msg] = openai_response_message
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=msg
                )
            ]

        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> NexusflowAIChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """

        import json
        import time
        has_system = False
        system_prompt_addition = "Use the chat() function to communicate the required information to the user or ask for more information. Make sure to use double quotes around the message."
        for data in openai_messages:
            if data["role"] == "tool":
                data["content"] = json.dumps(data["content"])
            if data["role"] == "system":
                has_system=True
                data["content"] = data["content"] + "\n\n" + system_prompt_addition

            if data["role"] == "assistant" and data["content"] in self.message_history:
                msg_reference = self.message_history[data["content"]].content
                reference = self.message_history[data["content"]].tool_calls[0]
                data["tool_calls"] = []
                data["content"] = msg_reference
                data["tool_calls"].append({
                    "id": reference.id,
                    "function": {
                        "arguments": reference.function.arguments,
                        "name": reference.function.name
                    },
                    "type": reference.type
                })

        if not has_system:
            openai_messages = [{"role" : "system", "content" : system_prompt_addition}] + openai_messages

        with all_logging_disabled():
            cc = self.nf_client.completions.create_with_tools(
                model=self.api_model_name,
                messages=cast(List[ChatCompletionMessageParam], openai_messages),
                tools=cast(List[ChatCompletionToolParam], openai_tools),
                temperature=0.7,
                max_tokens=4096
            )
            for choice in cc.choices:
                if choice.message.tool_calls is not None:
                    for tool_call in choice.message.tool_calls:
                        tool_call.id = tool_call.id.replace("-", "_")

            return cc


class AtheneV2Agent_Agent(AtheneAPIAgent):
    model_name = "Athene-V2-Agent"
    base_url = os.getenv("ATHENE_V2_AGENT_ENDPOINT")
    api_key = os.getenv("ATHENE_V2_AGENT_API_KEY")
    api_model_name = "Nexusflow/Athene-V2-Agent"
