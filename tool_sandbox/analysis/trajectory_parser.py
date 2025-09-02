"""Functions for parsing ToolSandbox trajectories into goal inference optimized format."""

import json
from pathlib import Path
from typing import Any

import polars as pl

from tool_sandbox.analysis.trajectory_types import (
    GoalInferenceToolCall,
    GoalInferenceTrajectory,
    ToolCallStep,
)
from tool_sandbox.common.execution_context import DatabaseNamespace
from tool_sandbox.common.tool_conversion import (
    tool_call_to_python_string,
)


def parse_trajectory_for_goal_inference(trajectory_path: str) -> GoalInferenceTrajectory:
    """Parse a complete trajectory folder for goal inference research.

    Args:
        trajectory_path: Path to trajectory folder containing execution_context.json

    Returns:
        Parsed trajectory with tool calls and database state changes

    Raises:
        FileNotFoundError: If execution_context.json is missing
        ValueError: If trajectory files contain invalid data
    """
    trajectory_dir = Path(trajectory_path)

    # Load execution context file (only file needed)
    execution_context_file = trajectory_dir / "execution_context.json"

    if not execution_context_file.exists():
        raise FileNotFoundError(f"execution_context.json not found in {trajectory_path}")

    with execution_context_file.open() as f:
        execution_context_data = json.load(f)

    # Find conversation boundaries from SANDBOX database
    conversation_start_index, conversation_end_index = _find_conversation_boundaries(execution_context_data)

    # Preprocess database changes into changes_dict
    database_changes, initial_states = _preprocess_database_changes(execution_context_data, conversation_start_index)

    # Extract tool calls from SANDBOX database only
    tool_calls = _extract_tool_calls_from_sandbox(
        execution_context_data, conversation_start_index, conversation_end_index
    )

    # Create step-by-step trajectory with simple dictionary lookup
    steps = _correlate_via_lookup(tool_calls, database_changes)

    return GoalInferenceTrajectory(
        steps=steps,
        scenario_name=trajectory_dir.name,
        initial_database_state=initial_states,
    )


def _extract_tool_calls_from_sandbox(
    execution_context_data: dict[str, Any], conversation_start_index: int, conversation_end_index: int
) -> list[GoalInferenceToolCall]:
    """Extract tool calls from SANDBOX database within conversation boundaries.

    Args:
        execution_context_data: Execution context data from execution_context.json
        conversation_start_index: Start of actual conversation
        conversation_end_index: End of actual conversation

    Returns:
        List of tool calls with exact message indices from SANDBOX
    """
    sandbox_records = execution_context_data.get("_dbs", {}).get("SANDBOX", [])
    tool_calls = []
    sequence_index = 0

    for record in sandbox_records:
        msg_idx = record.get("sandbox_message_index", 0)
        sender = record.get("sender")
        recipient = record.get("recipient")
        function_name = record.get("openai_function_name")
        content = record.get("content", "")
        call_id = record.get("openai_tool_call_id")

        # Only look at records within conversation boundaries
        if not (conversation_start_index <= msg_idx <= conversation_end_index):
            continue

        # Look for Agent tool calls to execution environment
        if sender == "AGENT" and recipient == "EXECUTION_ENVIRONMENT" and function_name and call_id:
            # Parse arguments from content (they're in the format: call_ID_parameters = {...})
            try:
                # Extract parameters from content
                lines = content.strip().split("\n")
                params_line = next(line for line in lines if "_parameters = " in line)
                params_str = params_line.split("_parameters = ", 1)[1]
                arguments = eval(params_str)  # Safe since this is our own data format
            except (IndexError, ValueError, SyntaxError):
                arguments = {}

            # Find corresponding result from execution environment
            result = None
            for result_record in sandbox_records:
                if (
                    result_record.get("sender") == "EXECUTION_ENVIRONMENT"
                    and result_record.get("recipient") == "AGENT"
                    and result_record.get("openai_tool_call_id") == call_id
                ):
                    result = result_record.get("content")
                    break

            # Create Python function string
            python_function_string = tool_call_to_python_string(function_name, arguments)

            tool_calls.append(
                GoalInferenceToolCall(
                    tool_name=function_name,
                    arguments=arguments,
                    result=result,
                    call_id=call_id,
                    sequence_index=sequence_index,
                    message_index=msg_idx,
                    python_function_string=python_function_string,
                )
            )

            sequence_index += 1

    return tool_calls


def _get_primary_key_column(namespace_name: str) -> str | None:
    """Get primary key column name for a database namespace.

    Args:
        namespace_name: Database namespace name

    Returns:
        Primary key column name or None if not applicable
    """
    # Define primary key fields for each namespace
    primary_key_fields = {
        "CONTACT": "person_id",
        "MESSAGING": "message_id",
        "REMINDER": "reminder_id",
        "SETTING": "device_id",
        "SANDBOX": None,  # No need to diff SANDBOX
    }

    return primary_key_fields.get(namespace_name)


def _find_conversation_boundaries(execution_context_data: dict[str, Any]) -> tuple[int, int]:
    """Find conversation start and end boundaries from SANDBOX database.

    Look for first and last end_conversation() calls to identify the main conversation.

    Args:
        execution_context_data: Execution context data from execution_context.json

    Returns:
        Tuple of (conversation_start_index, conversation_end_index)
    """
    sandbox_records = execution_context_data.get("_dbs", {}).get("SANDBOX", [])
    if not sandbox_records:
        return 0, 0

    end_conversation_indices = []

    for record in sandbox_records:
        sender = record.get("sender")
        content = record.get("content", "")
        msg_idx = record.get("sandbox_message_index", 0)

        # Look for USER calling end_conversation
        if sender == "USER" and "end_conversation()" in str(content):
            end_conversation_indices.append(msg_idx)

    if not end_conversation_indices:
        # No end_conversation found, return full range
        return 0, max(r.get("sandbox_message_index", 0) for r in sandbox_records)

    # Conversation starts after FIRST end_conversation
    # Conversation ends before LAST end_conversation (exclusive)
    conversation_start = end_conversation_indices[0] + 1
    if len(end_conversation_indices) > 1:
        conversation_end = end_conversation_indices[-1] - 1
    else:
        # Only one end_conversation, use max message index
        conversation_end = max(r.get("sandbox_message_index", 0) for r in sandbox_records)

    return conversation_start, conversation_end


def _preprocess_database_changes(
    execution_context_data: dict[str, Any], conversation_start_index: int
) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]]:
    """Preprocess database changes into a changes_dict for easy lookup.

    Args:
        execution_context_data: Execution context data from execution_context.json
        conversation_start_index: Where the actual conversation starts

    Returns:
        Tuple of (database_changes_dict, initial_states_dict)
        - database_changes_dict: {namespace: {message_index: [new_records]}}
        - initial_states_dict: {namespace: [initial_records]}
    """
    database_changes: dict[str, dict[int, list[dict[str, Any]]]] = {}
    initial_states: dict[str, list[dict[str, Any]]] = {}
    dbs = execution_context_data.get("_dbs", {})

    for namespace_name, records in dbs.items():
        if not records or namespace_name == "SANDBOX":
            continue  # Skip SANDBOX and empty databases

        try:
            # Convert string namespace to enum for validation
            DatabaseNamespace[namespace_name]
        except KeyError:
            continue  # Skip unknown namespaces

        df = _prepare_dataframe(records)
        if df is None:
            continue

        message_indexes = df["sandbox_message_index"].unique().sort().to_list()
        changes, initial = _process_namespace_changes(df, message_indexes, conversation_start_index, namespace_name)

        if changes:
            database_changes[namespace_name] = changes
        if initial:
            initial_states[namespace_name] = initial

    return database_changes, initial_states


def _prepare_dataframe(records: list[dict[str, Any]]) -> pl.DataFrame | None:
    """Prepare DataFrame from records, filtering out headguard rows."""
    df = pl.DataFrame(records)

    # Remove headguard rows (all null except sandbox_message_index)
    non_null_cols = [col for col in df.columns if col != "sandbox_message_index"]
    if non_null_cols:
        df = df.filter(pl.any_horizontal([pl.col(col).is_not_null() for col in non_null_cols]))

    return df if not df.is_empty() else None


def _process_namespace_changes(
    df: pl.DataFrame, message_indexes: list[int], conversation_start_index: int, namespace_name: str
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Process changes for a single namespace."""
    changes: dict[int, list[dict[str, Any]]] = {}
    initial_state_records = []
    accumulated_records = []

    for msg_idx in message_indexes:
        current_records = df.filter(pl.col("sandbox_message_index") == msg_idx).to_dicts()

        if msg_idx <= conversation_start_index:
            initial_state_records = current_records.copy()
            accumulated_records = current_records.copy()
        else:
            new_records = _calculate_new_records(current_records, accumulated_records, namespace_name, df)

            if new_records:
                changes[msg_idx] = new_records

            accumulated_records.extend(new_records)

    return changes, initial_state_records


def _calculate_new_records(
    current_records: list[dict[str, Any]],
    accumulated_records: list[dict[str, Any]],
    namespace_name: str,
    df: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Calculate new records by comparing current with accumulated."""
    pk_col = _get_primary_key_column(namespace_name)

    if pk_col and pk_col in df.columns:
        accumulated_pks = {r[pk_col] for r in accumulated_records if pk_col in r}
        return [r for r in current_records if r.get(pk_col) not in accumulated_pks]
    else:
        # For tables without primary keys, assume all current records are new
        return current_records


def _correlate_via_lookup(
    tool_calls: list[GoalInferenceToolCall], database_changes: dict[str, dict[int, list[dict[str, Any]]]]
) -> list[ToolCallStep]:
    """Correlate tool calls with database changes via simple dictionary lookup.

    Args:
        tool_calls: List of tool calls from SANDBOX database
        database_changes: Preprocessed changes dict {namespace: {message_index: [changes]}}

    Returns:
        List of ToolCallStep objects with exact database changes per tool call
    """
    steps = []

    for tool_call in tool_calls:
        # Direct lookup of database changes at this tool call's message index
        step_db_changes: dict[str, list[dict[str, Any]]] = {}

        for namespace_name, changes_by_index in database_changes.items():
            # Look up changes at tool_call message index + 1 (database changes happen after tool execution)
            changes_at_index = changes_by_index.get(tool_call.message_index + 1, [])
            if changes_at_index:
                step_db_changes[namespace_name] = changes_at_index

        steps.append(ToolCallStep(tool_call=tool_call, database_changes=step_db_changes))

    return steps
