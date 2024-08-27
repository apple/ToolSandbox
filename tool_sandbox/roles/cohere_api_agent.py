# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for the Cohere models using Cohere tool use API."""

import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from cohere import Client
from cohere.core import ApiError
from cohere.types import (
    Message as CohereMessage,
)
from cohere.types import (
    Message_Chatbot,
    Message_System,
    Message_Tool,
    Message_User,
    NonStreamedChatResponse,
    Tool,
    ToolCall,
    ToolParameterDefinitionsValue,
    ToolResult,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
)
from tool_sandbox.common.tool_conversion import (
    convert_to_openai_tool,
)
from tool_sandbox.roles.base_role import BaseRole

LOGGER = logging.getLogger(__name__)


class CohereAPIAgent(BaseRole):
    """Agent which uses the Cohere API and models."""

    role_type: RoleType = RoleType.AGENT
    model_name: str
    client: Client

    tool_calls: Dict[str, Tuple[str, ToolCall]] = {}
    """Stores previous tool calls by id, so that the prompt can be recreated on later turns"""

    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name
        # The Cohere SDK will look for an api key in the `CO_API_KEY` environment variable.
        self.client = Client()

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type.
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses.

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return

        available_tools: Dict[str, Callable[..., Any]] = {}
        if (
            messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
        ):
            available_tools = self.get_available_tools()

        # Call model
        response_messages = self.model_inference(
            messages=messages,
            available_tools=available_tools,
            agent_to_execution_facing_tool_name=get_current_context().get_agent_to_execution_facing_tool_name(),
        )

        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ApiError),
    )
    def model_inference(
        self,
        messages: List[Message],
        available_tools: Dict[str, Callable[..., Any]],
        agent_to_execution_facing_tool_name: Dict[str, str],
    ) -> List[Message]:
        """Run Cohere model inference.

        Args:
            messages: Messages in tool sandbox format to send to the LLM.
            available_tools: The tools that are available to the agent.
            agent_to_execution_facing_tool_name: A mapping between the (potentially scrambled) tool name used by the
             agent, and the unscrambled tool name used by the Python interpreter.

        Returns:
            A list of tool sandbox messages.

        Raises:
            AssertionError: When the messages are invalid.
        """
        assert len(messages) >= 1, "chat history must be at least one length"
        cohere_messages = to_cohere_messages(
            messages=messages, previous_tool_calls=self.tool_calls
        )

        cohere_tools = [
            to_cohere_tool(name, tool) for name, tool in available_tools.items()
        ]

        cohere_response = self.client.chat(
            tools=cohere_tools,
            chat_history=cohere_messages[:-1],
            message=cohere_messages[-1].message
            if isinstance(cohere_messages[-1], Message_User)
            else "",
            tool_results=cohere_messages[-1].tool_results
            if isinstance(cohere_messages[-1], Message_Tool)
            else None,
            model=self.model_name,
            temperature=0.0,
        )

        return self.response_to_messages(
            response=cohere_response,
            sender=self.role_type,
            available_tool_names=set(available_tools.keys()),
            agent_to_execution_facing_tool_name=agent_to_execution_facing_tool_name,
        )

    def response_to_messages(
        self,
        response: NonStreamedChatResponse,
        sender: RoleType,
        available_tool_names: Set[str],
        agent_to_execution_facing_tool_name: Dict[str, str],
    ) -> list[Message]:
        if response.tool_calls:
            messages: list[Message] = []
            for tool_call in response.tool_calls:
                tool_call_id = str(uuid.uuid4())
                self.tool_calls[tool_call_id] = (response.text, tool_call)
                messages.append(
                    Message(
                        sender=sender,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=tool_call_to_python_code(
                            execution_facing_tool_name=agent_to_execution_facing_tool_name[
                                tool_call.name
                            ],
                            tool_call=tool_call,
                            available_tool_names=available_tool_names,
                        ),
                        openai_tool_call_id=tool_call_id,
                        openai_function_name=tool_call.name,
                    )
                )

            return messages
        else:
            return [
                Message(
                    sender=sender,
                    recipient=RoleType.USER,
                    content=response.text,
                )
            ]


def to_cohere_tool(name: str, tool: Callable[..., Any]) -> Tool:
    """Converts a tool sandbox tool into a Cohere API tool.
    Args:
        name: The name of the tool.
        tool: The definition of the tool.

    Returns:
        A Cohere API tool.
    """
    formatted = convert_to_openai_tool(tool, name=name)["function"]
    json_to_python_types = {
        "string": "str",
        "number": "float",
        "boolean": "bool",
        "integer": "int",
        "array": "List",
        "object": "Dict",
    }

    parameters = formatted.get("parameters", {})

    return Tool(
        name=formatted.get("name"),
        description=formatted.get("description"),
        parameter_definitions={
            param_name: ToolParameterDefinitionsValue(
                description=param_definition.get("description"),
                type=json_to_python_types.get(param_definition.get("type")) or "str",
                required=param_name in parameters.get("required", []),
            )
            for param_name, param_definition in parameters.get("properties", {}).items()
        },
    )


def tool_call_to_python_code(
    execution_facing_tool_name: str,
    tool_call: ToolCall,
    available_tool_names: set[str],
) -> str:
    """Converts a Cohere tool call into Python code for calling the function.
    Args:
        execution_facing_tool_name:  The execution facing name of the function. In the
                                     case of tool name scrambling the Cohere API in- and
                                     outputs are filled with scrambled tool names. When
                                     executing the code we need to use the actual tool
                                     name.
        tool_call:                   The Cohere tool call describing the function name and arguments.
        available_tool_names:        Set of available tools.

    Returns:
        The Python code for making the tool call.

    Raises:
        KeyError: If the selected tool is not a known tool.
    """
    agent_facing_tool_name = tool_call.name
    if agent_facing_tool_name not in available_tool_names:
        raise KeyError(
            f"Agent tool call {agent_facing_tool_name=} is not a known allowed tool. Options "
            f"are {available_tool_names=}."
        )

    function_call_code = (
        f"{agent_facing_tool_name}_parameters = {tool_call.parameters}\n"
        f"{agent_facing_tool_name}_response = {execution_facing_tool_name}(**{agent_facing_tool_name}_parameters)\n"
        f"print(repr({agent_facing_tool_name}_response))"
    )
    return function_call_code


def to_cohere_messages(
    messages: List[Message], previous_tool_calls: Dict[str, Tuple[str, ToolCall]]
) -> List[CohereMessage]:
    """Converts a list of tool sandbox messages into Cohere API messages.

    Args:
        messages: List of tool sandbox messages.
        previous_tool_calls: Previous Cohere tool calls.

    Returns:
        A list of Cohere API messages and, optionally, a string to use a system prompt.
    """
    cohere_messages: List[CohereMessage] = []

    for message in messages:
        if message.sender == RoleType.SYSTEM and message.recipient == RoleType.AGENT:
            cohere_messages.append(Message_System(message=message.content))
        elif message.sender == RoleType.USER and message.recipient == RoleType.AGENT:
            cohere_messages.append(Message_User(message=message.content))
        elif (
            message.sender == RoleType.EXECUTION_ENVIRONMENT
            and message.recipient == RoleType.AGENT
        ):
            assert (
                message.openai_tool_call_id is not None
            ), "openai_tool_call_id should be provided"
            cohere_messages.append(
                Message_Tool(
                    tool_results=[
                        ToolResult(
                            call=previous_tool_calls[message.openai_tool_call_id][1],
                            outputs=[{"output": message.content}],
                        )
                    ]
                )
            )
        elif (
            message.sender == RoleType.AGENT
            and message.recipient == RoleType.EXECUTION_ENVIRONMENT
        ):
            assert (
                message.openai_tool_call_id is not None
            ), "openai_tool_call_id should be provided"
            tool_call_message, tool_call = previous_tool_calls[
                message.openai_tool_call_id
            ]
            cohere_messages.append(
                Message_Chatbot(
                    message=tool_call_message,
                    tool_calls=[tool_call],
                )
            )
        elif message.sender == RoleType.AGENT and message.recipient == RoleType.USER:
            cohere_messages.append(Message_Chatbot(message=message.content))
        else:
            raise ValueError(
                "Unrecognized sender recipient pair "
                f"{(message.sender, message.recipient)}"
            )

    return cohere_messages
