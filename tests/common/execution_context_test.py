# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for tool_sandbox.common.execution_context"""

import code
import copy

import polars as pl
import pytest

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    ScenarioCategories,
    get_current_context,
    new_context,
    new_context_with_attribute,
    set_current_context,
)
from tool_sandbox.common.tool_discovery import ToolBackend


@pytest.fixture
def default_execution_context() -> ExecutionContext:
    """Default init ExecutionContext

    Returns:
        A default init ExecutionContext object
    """
    return ExecutionContext()


@pytest.fixture
def populated_execution_context() -> ExecutionContext:
    """Execution context with a few entries populated in SETTINGS database and SANDBOX database

    Returns:
        A default init ExecutionContext object
    """
    test_context: ExecutionContext = ExecutionContext()
    # Add 3 entries into SANDBOX, remember there's a headguard for each snapshot
    test_context._dbs[DatabaseNamespace.SANDBOX] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 0,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.AGENT,
                "content": "Be Good",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 1,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 1,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.USER,
                "content": "Be Good",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 2,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 2,
                "sender": RoleType.SYSTEM,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": "import json\n",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 3,
                "sender": None,
                "recipient": None,
                "content": None,
                "conversation_active": None,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            {
                "sandbox_message_index": 3,
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Hi",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
    )

    test_context._dbs[DatabaseNamespace.SETTING] = pl.DataFrame(
        [
            {
                "sandbox_message_index": 0,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "latitude": None,
                "longitude": None,
            },
            {
                "sandbox_message_index": 1,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "latitude": None,
                "longitude": None,
            },
            {
                "sandbox_message_index": 1,
                "device_id": "1",
                "cellular": True,
                "wifi": False,
                "location_service": True,
                "low_battery_mode": False,
                "latitude": 0,
                "longitude": 0,
            },
            {
                "sandbox_message_index": 3,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "latitude": None,
                "longitude": None,
            },
            {
                "sandbox_message_index": 3,
                "device_id": "1",
                "cellular": False,
                "wifi": False,
                "location_service": True,
                "low_battery_mode": False,
                "latitude": 0,
                "longitude": 0,
            },
        ],
        schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
    )
    # Add other attributes
    test_context.tool_allow_list = ["end_conversation"]
    test_context.tool_deny_list = ["get_current_location"]
    test_context.trace_tool = False
    test_context.tool_augmentation_list = [ScenarioCategories.TOOL_NAME_SCRAMBLED]
    test_context.preferred_tool_backend = ToolBackend.DEFAULT
    command = code.compile_command("a=1", symbol="exec")
    assert command is not None
    test_context.interactive_console.runcode(command)

    return test_context


def test_drop_headguard() -> None:
    assert ExecutionContext.drop_headguard(
        pl.DataFrame({"sandbox_message_index": 0, "a": None, "b": None})
    ).is_empty()
    assert ExecutionContext.drop_headguard(
        pl.DataFrame({"a": None, "b": None})
    ).is_empty()


def test_max_sandbox_message_index(default_execution_context: ExecutionContext) -> None:
    # Empty SANDBOX database
    assert default_execution_context.max_sandbox_message_index == -1
    # Add an entry
    default_execution_context._dbs[DatabaseNamespace.SANDBOX] = (
        default_execution_context._dbs[DatabaseNamespace.SANDBOX].vstack(
            pl.DataFrame(
                {
                    "sandbox_message_index": 0,
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.AGENT,
                    "content": "Be Good",
                    "conversation_active": True,
                    "openai_tool_call_id": None,
                    "openai_function_name": None,
                    "tool_call_exception": None,
                    "tool_trace": None,
                    "visible_to": None,
                },
                schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
            )
        )
    )
    assert default_execution_context.max_sandbox_message_index == 0


def test_get_most_recent_snapshot_sandbox_message_index_empty(
    default_execution_context: ExecutionContext,
) -> None:
    assert (
        default_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.MESSAGING, query_index=1
        )
        == -1
    )


def test_get_most_recent_snapshot_sandbox_message_index_populated(
    populated_execution_context: ExecutionContext,
) -> None:
    # Out of bounds
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=0
        )
        == 0
    )
    # Bisect
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=2
        )
        == 1
    )
    # Max allowed
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=4
        )
        == 3
    )
    # Error
    with pytest.raises(IndexError):
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=5
        )


def test_maybe_create_snapshot(populated_execution_context: ExecutionContext) -> None:
    settings_database = copy.deepcopy(
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
    )
    populated_execution_context._maybe_create_snapshot(
        namespace=DatabaseNamespace.SETTING
    )
    # New snapshot should exist, matching previous latest exactly except index
    assert (
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
        .filter(pl.col("sandbox_message_index") == 3)
        .drop("sandbox_message_index")
        .equals(
            populated_execution_context._dbs[DatabaseNamespace.SETTING]
            .filter(pl.col("sandbox_message_index") == 4)
            .drop("sandbox_message_index")
        )
    )
    # Previous latest should not be modified
    assert (
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
        .filter(pl.col("sandbox_message_index") <= 3)
        .equals(settings_database)
    )
    # New snapshots should not be created after the snapshot for max_sandbox_message_index + 1 exists
    settings_database_with_snapshot = copy.deepcopy(
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
    )
    populated_execution_context._maybe_create_snapshot(
        namespace=DatabaseNamespace.SETTING
    )
    assert populated_execution_context._dbs[DatabaseNamespace.SETTING].equals(
        settings_database_with_snapshot
    )


def test_get_database(populated_execution_context: ExecutionContext) -> None:
    # Default values
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING]
            .filter(pl.col("sandbox_message_index") == 3)
            .drop("sandbox_message_index")
        )
    )
    # Don't drop index
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING, drop_sandbox_message_index=False
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
                pl.col("sandbox_message_index") == 3
            )
        )
    )
    # Earlier index
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        sandbox_message_index=2,
        drop_sandbox_message_index=False,
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
                pl.col("sandbox_message_index") == 1
            )
        )
    )
    # All history
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    ).equals(
        ExecutionContext.drop_headguard(
            populated_execution_context._dbs[DatabaseNamespace.SETTING]
        )
    )
    # Don't drop headguard
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING, drop_headguard=False
    ).equals(
        populated_execution_context._dbs[DatabaseNamespace.SETTING]
        .filter(pl.col("sandbox_message_index") == 3)
        .drop("sandbox_message_index")
    )


def test_add_to_database(populated_execution_context: ExecutionContext) -> None:
    # Incorrect column name
    with pytest.raises(KeyError):
        populated_execution_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX,
            rows=[
                {"wrong_name": "test"},
            ],
        )
    # Adding messages
    populated_execution_context.add_to_database(
        namespace=DatabaseNamespace.SANDBOX,
        rows=[
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": "Hey",
                "conversation_active": True,
            },
            {
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Howdy",
                "conversation_active": True,
            },
        ],
    )
    assert ExecutionContext.drop_headguard(
        populated_execution_context._dbs[DatabaseNamespace.SANDBOX].filter(
            (pl.col("sandbox_message_index") >= 4)
            & (pl.col("sandbox_message_index") <= 5)
        )
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 4,
                    "sender": RoleType.AGENT,
                    "recipient": RoleType.USER,
                    "content": "Hey",
                    "conversation_active": True,
                    "openai_tool_call_id": None,
                    "openai_function_name": None,
                    "tool_call_exception": None,
                    "tool_trace": None,
                },
                {
                    "sandbox_message_index": 5,
                    "sender": RoleType.USER,
                    "recipient": RoleType.AGENT,
                    "content": "Howdy",
                    "conversation_active": True,
                    "openai_tool_call_id": None,
                    "openai_function_name": None,
                    "tool_call_exception": None,
                    "tool_trace": None,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        )
    )
    # Adding to non SANDBOX database, snapshot should be created
    # Note that normally SETTINGS should only have 1 row in each snapshot. The following is just for testing purposes
    populated_execution_context.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "device_id": "1",
                "cellular": True,
                "wifi": False,
                "location_service": True,
                "low_battery_mode": False,
                "latitude": 0,
                "longitude": 0,
            }
        ],
    )
    assert ExecutionContext.drop_headguard(
        populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
            pl.col("sandbox_message_index") == 6
        )
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": False,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": 0,
                    "longitude": 0,
                },
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": 0,
                    "longitude": 0,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )
    # Adding to non SANDBOX database, snapshot should not be created
    populated_execution_context.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "device_id": "1",
                "cellular": True,
                "wifi": True,
                "location_service": True,
                "low_battery_mode": False,
                "latitude": 0,
                "longitude": 0,
            }
        ],
    )
    assert ExecutionContext.drop_headguard(
        populated_execution_context._dbs[DatabaseNamespace.SETTING].filter(
            pl.col("sandbox_message_index") == 6
        )
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": False,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": 0,
                    "longitude": 0,
                },
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": 0,
                    "longitude": 0,
                },
                {
                    "sandbox_message_index": 6,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": True,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": 0,
                    "longitude": 0,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )


def test_remove_from_database(populated_execution_context: ExecutionContext) -> None:
    # Remove from sandbox
    with pytest.raises(KeyError):
        populated_execution_context.remove_from_database(
            namespace=DatabaseNamespace.SANDBOX,
            predicate=pl.col("sandbox_message_index") == 0,
        )
    # Remove 1. Headguard should remain, get should show empty if headguard is dropped, snapshot ind should be 4
    populated_execution_context.remove_from_database(
        namespace=DatabaseNamespace.SETTING,
        predicate=(pl.col("cellular") == pl.lit(False)),
    )
    assert (
        populated_execution_context.get_most_recent_snapshot_sandbox_message_index(
            namespace=DatabaseNamespace.SETTING, query_index=4
        )
        == 4
    )
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING
    ).is_empty()
    assert populated_execution_context.get_database(
        namespace=DatabaseNamespace.SETTING,
        drop_headguard=False,
        drop_sandbox_message_index=False,
    ).equals(
        pl.DataFrame(
            {
                "sandbox_message_index": 4,
                "device_id": None,
                "cellular": None,
                "wifi": None,
                "location_service": None,
                "low_battery_mode": None,
                "latitude": None,
                "longitude": None,
            },
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SETTING],
        )
    )


def test_update_database(populated_execution_context: ExecutionContext) -> None:
    # Update sandbox. Should not create snapshot
    populated_execution_context.update_database(
        namespace=DatabaseNamespace.SANDBOX,
        dataframe=populated_execution_context.get_database(
            namespace=DatabaseNamespace.SANDBOX
        ).with_columns(pl.lit("Hey").alias("content")),
    )
    populated_execution_context.get_database(
        namespace=DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
    ).equals(
        pl.DataFrame(
            {
                "sandbox_message_index": 3,
                "sender": RoleType.USER,
                "recipient": RoleType.AGENT,
                "content": "Hey",
                "conversation_active": True,
                "openai_tool_call_id": None,
                "openai_function_name": None,
                "tool_call_exception": None,
                "tool_trace": None,
                "visible_to": None,
            },
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        )
    )
    # Update Setting. Should create snapshot. Should contain headguard
    populated_execution_context.update_database(
        namespace=DatabaseNamespace.SETTING,
        dataframe=populated_execution_context.get_database(
            namespace=DatabaseNamespace.SETTING
        ).with_columns(pl.lit(True).alias("cellular")),
    )
    populated_execution_context.get_database(
        namespace=DatabaseNamespace.SANDBOX,
        drop_sandbox_message_index=False,
        drop_headguard=False,
    ).equals(
        pl.DataFrame(
            [
                {
                    "sandbox_message_index": 4,
                    "device_id": None,
                    "cellular": None,
                    "wifi": None,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": None,
                    "longitude": None,
                },
                {
                    "sandbox_message_index": 4,
                    "device_id": "1",
                    "cellular": True,
                    "wifi": False,
                    "location_service": True,
                    "low_battery_mode": False,
                    "latitude": 0,
                    "longitude": 0,
                },
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        )
    )


def test_new_context(
    default_execution_context: ExecutionContext,
    populated_execution_context: ExecutionContext,
) -> None:
    set_current_context(default_execution_context)
    with new_context(populated_execution_context):
        assert (
            get_current_context()
            .get_database(DatabaseNamespace.SANDBOX)
            .equals(populated_execution_context.get_database(DatabaseNamespace.SANDBOX))
        )
    assert (
        get_current_context()
        .get_database(DatabaseNamespace.SANDBOX)
        .equals(default_execution_context.get_database(DatabaseNamespace.SANDBOX))
    )
    # Check release during exception
    with pytest.raises(RuntimeError):
        with new_context(populated_execution_context):
            raise RuntimeError()
    assert (
        get_current_context()
        .get_database(DatabaseNamespace.SANDBOX)
        .equals(default_execution_context.get_database(DatabaseNamespace.SANDBOX))
    )


def test_new_context_with_attribute(
    default_execution_context: ExecutionContext,
) -> None:
    set_current_context(default_execution_context)
    with new_context_with_attribute(trace_tool=True):
        assert get_current_context().trace_tool
    assert not get_current_context().trace_tool
    # Check release during exception
    with pytest.raises(RuntimeError):
        with new_context_with_attribute(trace_tool=True):
            raise RuntimeError()
    assert not get_current_context().trace_tool


def test_context_copying() -> None:
    context = ExecutionContext()

    # Set a local variable in the interactive console. We will use this variable to
    # ensure that we can create deep copies of the execution context. Also import a
    # module and a user-defined function so we can test these as well.
    console = context.interactive_console
    assert "a" not in console.locals
    console.runsource("a=0")
    assert console.locals["a"] == 0
    console.runsource("import math; b = math.degrees(math.pi)")
    assert pytest.approx(180.0) == console.locals["b"]
    console.runsource("from tool_sandbox.common.utils import deterministic_uuid")
    assert "deterministic_uuid" in console.locals

    # Create a copy and change the value of `a` to allow testing that we indeed created
    # a deep copy.
    clone = copy.deepcopy(context).interactive_console
    assert clone.locals["a"] == 0
    clone.runsource("a=1")
    assert clone.locals["a"] == 1
    # The variable in the original context should be unchanged.
    assert console.locals["a"] == 0

    # Test that the `math` module was correctly copied. `InteractiveConsole` internally
    # catches exceptions and just prints them so we use the existence of a variable as
    # a proxy for the statement being executed successfully. The lines below ensure that
    # this approach can indeed be used to test that statement execution failed.
    clone.runsource("c = json.dumps({})")
    assert "c" not in clone.locals

    # If for some reason the `math` import was not copied correctly then the next
    # statement would raise an exception that the `InteractiveConsole` would catch
    # internally. The variable `c` would then be undefined.
    clone.runsource("c = math.degrees(math.pi)")
    assert pytest.approx(console.locals["b"]) == clone.locals["c"]

    # Ensure that the user-defined `deterministic_uuid` function exists in the cloned
    # console and can be called.
    assert "deterministic_uuid" in clone.locals
    assert "d" not in clone.locals
    clone.runsource("d = deterministic_uuid(payload='test')")
    assert "d" in clone.locals


def test_serialization_copying(populated_execution_context: ExecutionContext) -> None:
    copied_context = ExecutionContext.from_dict(
        populated_execution_context.to_dict(serialize_console=True)
    )
    for namespace in DatabaseNamespace:
        assert copied_context._dbs[namespace].equals(
            populated_execution_context._dbs[namespace]
        )
    assert copied_context.tool_allow_list == populated_execution_context.tool_allow_list
    assert copied_context.trace_tool == populated_execution_context.trace_tool
    assert (
        copied_context.tool_augmentation_list
        == populated_execution_context.tool_augmentation_list
    )
    assert (
        copied_context.preferred_tool_backend
        == populated_execution_context.preferred_tool_backend
    )
    assert (
        populated_execution_context.interactive_console.locals["a"]
        == copied_context.interactive_console.locals["a"]
    )
    copied_context = ExecutionContext.from_dict(
        populated_execution_context.to_dict(serialize_console=False)
    )
    assert "a" not in copied_context.interactive_console.locals
