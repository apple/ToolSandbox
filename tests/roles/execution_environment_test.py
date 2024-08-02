# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for tool_sandbox.roles.execution_environment"""

import textwrap
from typing import Iterator

import polars as pl
import pytest

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    new_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.common.utils import deterministic_uuid
from tool_sandbox.roles.execution_environment import ExecutionEnvironment


@pytest.fixture(scope="function", autouse=True)
def execution_context() -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    with new_context(ExecutionContext()):
        yield


@pytest.fixture
def execution_environment() -> ExecutionEnvironment:
    """Execution environment object used for testing

    Returns:
        An execution environment object
    """
    return ExecutionEnvironment()


def test_execution_environment_syntax_error(
    execution_environment: ExecutionEnvironment,
) -> None:
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="improt math",
            )
        ]
    )
    execution_environment.respond()
    content = textwrap.dedent("""\
      File "<input>", line 1
        improt math
               ^
    SyntaxError: invalid syntax
    """)
    assert execution_environment.get_messages()[-1] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=content,
        conversation_active=True,
        tool_call_exception=content,
    )


def test_execution_environment_raise_error(
    execution_environment: ExecutionEnvironment,
) -> None:
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content='raise ValueError("This is a test error")',
            )
        ]
    )
    execution_environment.respond()
    content = "ValueError: This is a test error"
    assert execution_environment.get_messages()[-1] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=content,
        conversation_active=True,
        tool_call_exception=content,
    )


def test_execution_environment_incomplete_command(
    execution_environment: ExecutionEnvironment,
) -> None:
    message = Message(
        sender=RoleType.AGENT,
        recipient=RoleType.EXECUTION_ENVIRONMENT,
        content="if True:",  # < incomplete code
    )
    execution_environment.add_messages([message])
    execution_environment.respond()
    content = (
        "Error: The given code was incomplete and could not be executed: "
        f"'{message.content}'"
    )
    assert execution_environment.get_messages()[-1] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=content,
        conversation_active=True,
        tool_call_exception=content,
    )


def test_execution_environment_successful_execution(
    execution_environment: ExecutionEnvironment,
) -> None:
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="a = 1\nb = 2\nc = a + b\nprint(c)",
            )
        ]
    )
    execution_environment.respond()
    content = "3"
    assert execution_environment.get_messages()[-1] == Message(
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        content=textwrap.dedent(content),
        conversation_active=True,
    )


def test_valid_parallel_tool_call(
    execution_environment: ExecutionEnvironment,
) -> None:
    # Set up the environment by adding a system message that specifies which tools to
    # import.
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    "import json\n"
                    "from tool_sandbox.tools.contact import search_contacts\n"
                ),
                conversation_active=True,
            )
        ]
    )
    execution_context = get_current_context()
    execution_context.add_to_database(
        DatabaseNamespace.CONTACT,
        rows=[
            {
                "person_id": deterministic_uuid(payload="Tomas Haake"),
                "name": "Tomas Haake",
                "phone_number": "+11233344455",
                "relationship": "self",
                "is_self": True,
            },
            {
                "person_id": deterministic_uuid(payload="Fredrik Thordendal"),
                "name": "Fredrik Thordendal",
                "phone_number": "+12453344098",
                "relationship": "friend",
                "is_self": False,
            },
        ],
    )
    execution_environment.respond()

    # Agents can issue parallel tool calls. The expectation is that these are
    # independent and can thus be executed in any order. This is true for the tool calls
    # defined below.
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters = {'name': 'Tomas Haake'}\ncall_ucWrTNIM4Prh1KlK77k1Zjeu_response = search_contacts(**call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters)\nprint(repr(call_ucWrTNIM4Prh1KlK77k1Zjeu_response))",
                conversation_active=True,
                openai_tool_call_id="call_ucWrTNIM4Prh1KlK77k1Zjeu",
                openai_function_name="search_contacts",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_e60fwDpTW2yecfBg2BQPNy88_parameters = {'name': 'Fredrik Thordendal'}\ncall_e60fwDpTW2yecfBg2BQPNy88_response = search_contacts(**call_e60fwDpTW2yecfBg2BQPNy88_parameters)\nprint(repr(call_e60fwDpTW2yecfBg2BQPNy88_response))",
                conversation_active=True,
                openai_tool_call_id="call_e60fwDpTW2yecfBg2BQPNy88",
                openai_function_name="search_contacts",
            ),
        ]
    )

    execution_environment.respond()

    # Without setting `get_all_history_snapshots` we only get the latest tool call back.
    df = get_current_context().get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
    )
    assert RoleType.EXECUTION_ENVIRONMENT == df[-2]["sender"][0]
    assert RoleType.AGENT == df[-2]["recipient"][0]
    assert "search_contacts" == df[-2]["openai_function_name"][0]
    # If all possible orderings of the tool calls succeed then the response should match
    # the original requests.
    assert "Tomas Haake" in df[-2]["content"][0]

    assert RoleType.EXECUTION_ENVIRONMENT == df[-1]["sender"][0]
    assert RoleType.AGENT == df[-1]["recipient"][0]
    assert "search_contacts" == df[-1]["openai_function_name"][0]
    assert "Fredrik Thordendal" in df[-1]["content"][0]


def test_invalid_parallel_tool_call(
    execution_environment: ExecutionEnvironment,
) -> None:
    # Set up the environment by adding a system message that specifies which tools to
    # import.
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=(
                    "import json\n"
                    "from tool_sandbox.tools.setting import set_wifi_status\n"
                    "from tool_sandbox.tools.rapid_api_search_tools import search_stock\n"
                ),
                conversation_active=True,
            )
        ]
    )
    execution_context = get_current_context()
    settings_df = execution_context.get_database(DatabaseNamespace.SETTING)
    execution_context.update_database(
        DatabaseNamespace.SETTING, settings_df.with_columns(pl.lit(False).alias("wifi"))
    )
    execution_environment.respond()

    # Agents can issue parallel tool calls. The expectation is that these are
    # independent and can thus be executed in any order. If that is not the case it
    # means that the tool calls are not truly parallel, but have some dependency on each
    # other. In the example below the tool calls depend on each other, but when executed
    # sequentially in the given order they would succeed. Since the execution
    # environment calls the tools in all order permutations the overall result should be
    # a failure. More specifically, when calling the `search_stock` tool first it will
    # fail with an exception saying that WiFi is not enabled.
    execution_environment.add_messages(
        [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters = {'on': True}\ncall_ucWrTNIM4Prh1KlK77k1Zjeu_response = set_wifi_status(**call_ucWrTNIM4Prh1KlK77k1Zjeu_parameters)\nprint(repr(call_ucWrTNIM4Prh1KlK77k1Zjeu_response))",
                conversation_active=True,
                openai_tool_call_id="call_ucWrTNIM4Prh1KlK77k1Zjeu",
                openai_function_name="set_wifi_status",
            ),
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content="call_e60fwDpTW2yecfBg2BQPNy88_parameters = {'query': 'Apple'}\ncall_e60fwDpTW2yecfBg2BQPNy88_response = search_stock(**call_e60fwDpTW2yecfBg2BQPNy88_parameters)\nprint(repr(call_e60fwDpTW2yecfBg2BQPNy88_response))",
                conversation_active=True,
                openai_tool_call_id="call_e60fwDpTW2yecfBg2BQPNy88",
                openai_function_name="search_stock",
            ),
        ]
    )

    execution_environment.respond()
    # WiFi should still be off.
    assert not execution_context.get_database(DatabaseNamespace.SETTING)["wifi"][0]

    # Without setting `get_all_history_snapshots` we only get the latest tool call back.
    df = get_current_context().get_database(
        DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
    )
    # The response should be in the same order as the (potentially permutated) requests.
    assert RoleType.EXECUTION_ENVIRONMENT == df[-2]["sender"][0]
    assert RoleType.AGENT == df[-2]["recipient"][0]
    assert "search_stock" == df[-2]["openai_function_name"][0]
    assert df[-2]["tool_call_exception"][0] is not None

    assert RoleType.EXECUTION_ENVIRONMENT == df[-1]["sender"][0]
    assert RoleType.AGENT == df[-1]["recipient"][0]
    assert "set_wifi_status" == df[-1]["openai_function_name"][0]
    assert "None" in df[-1]["content"][0]
