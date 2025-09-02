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
class GoalInferencePrediction:
    """Represents a model's prediction at a specific step in trajectory."""

    prediction_type: str  # "wait" or "goal"
    content: str
    model_response: str  # Raw model output


@dataclass
class ToolCallStep:
    """Represents a single step in the trajectory: tool call + its database effects.

    This structure directly links each tool call to the database changes it caused,
    which is essential for goal inference research.
    """

    tool_call: GoalInferenceToolCall
    database_changes: dict[str, list[dict[str, Any]]]  # namespace -> list of changed records
    prediction: GoalInferencePrediction | None = None  # Model's prediction at this step


@dataclass
class GoalInferenceTrajectory:
    """Complete trajectory with goal inference predictions at each step.

    Contains step-by-step progression where each step links a tool call to its
    database effects and optionally includes model predictions.
    """

    steps: list[ToolCallStep]  # Each step = tool call + database effects + optional prediction
    scenario_name: str
    initial_database_state: dict[str, list[dict[str, Any]]]  # Complete initial state for context
    model_name: str | None = None  # Model used for predictions (if any)
    final_prediction: str | None = None  # Last non-wait prediction
