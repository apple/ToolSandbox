"""Data structures optimized for goal inference from ToolSandbox trajectories."""

from dataclasses import dataclass
from typing import Any

from tool_sandbox.common.execution_context import DatabaseNamespace


@dataclass
class GoalInferenceToolCall:
    """Represents a tool call optimized for goal inference analysis.

    This structure provides clean access to tool call information needed for
    goal inference models, including Python function string representation.
    """

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    call_id: str
    sequence_index: int  # Order in conversation
    message_index: int  # sandbox_message_index from execution context
    python_function_string: str  # e.g., "search_contacts(name='Alex')"


@dataclass
class DatabaseStateSnapshot:
    """Represents database state at a specific point in trajectory execution.

    Reuses existing DatabaseNamespace structure while providing goal inference
    context for how tool calls affect the world state.
    """

    namespace: DatabaseNamespace
    data: list[dict[str, Any]]
    message_index: int


@dataclass
class ToolCallStep:
    """Represents a single step in the trajectory: tool call + its database effects.

    This structure directly links each tool call to the database changes it caused,
    which is essential for goal inference research.
    """

    tool_call: GoalInferenceToolCall
    database_changes: dict[str, list[dict[str, Any]]]  # namespace -> list of changed records


@dataclass
class ParsedGoalInferenceTrajectory:
    """Complete parsed trajectory optimized for goal inference research.

    Contains step-by-step progression where each step links a tool call to its
    database effects, formatted for goal inference model consumption.
    """

    steps: list[ToolCallStep]  # Each step = tool call + its database effects
    scenario_name: str
    initial_database_state: dict[str, list[dict[str, Any]]]  # Complete initial state for context
