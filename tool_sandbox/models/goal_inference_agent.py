"""Goal inference agent for analyzing tool usage patterns."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from tool_sandbox.analysis.trajectory_parser import parse_trajectory_for_goal_inference
from tool_sandbox.analysis.trajectory_types import (
    GoalInferencePrediction,
    GoalInferenceTrajectory,
    ToolCallStep,
)
from tool_sandbox.configs.models import APIModelConfig
from tool_sandbox.models.litellm import LiteLLMModel
from tool_sandbox.models.parsers import BoxedGoalInferenceParser

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse


class GoalInferenceAgent:
    """Agent for running goal inference experiments on trajectories."""

    def __init__(self, model_name: str, config: Optional[APIModelConfig] = None) -> None:
        """Initialize the goal inference agent.

        Args:
            model_name: Name of the LiteLLM model to use
            config: Optional model configuration
        """
        self.logger = logging.getLogger(__name__)

        if config is None:
            config = APIModelConfig()
        config.model_name = model_name

        self.model_name = model_name
        self.llm = LiteLLMModel(config=config, name=f"GoalInference-{model_name}", logger=self.logger)
        self.parser = BoxedGoalInferenceParser()

        # Load system prompt
        prompts_file = Path(__file__).parent / "goal_inference_prompts.yaml"
        with prompts_file.open() as f:
            prompts: dict[str, str] = yaml.safe_load(f)
            self.system_prompt = prompts["system_prompt"]
            self.task_prompt = prompts["task_prompt"]

    def run_goal_inference_on_trajectory_path(self, trajectory_path: str) -> GoalInferenceTrajectory:
        """Run goal inference on a trajectory directory.

        Args:
            trajectory_path: Path to trajectory directory

        Returns:
            Complete trajectory with predictions
        """
        trajectory = parse_trajectory_for_goal_inference(trajectory_path)
        return self.run_goal_inference(trajectory)

    def run_goal_inference(self, trajectory: GoalInferenceTrajectory) -> GoalInferenceTrajectory:
        """Run goal inference on a parsed trajectory.

        Args:
            trajectory: Parsed trajectory to analyze

        Returns:
            Complete trajectory with predictions added to each step
        """
        conversation_messages = [{"role": "system", "content": self.system_prompt}]
        final_prediction = None

        for step_idx, step in enumerate(trajectory.steps):
            # Format this step for the model
            step_message = self._format_step_message(step, step_idx + 1)
            conversation_messages.append({"role": "user", "content": step_message})

            # Get model prediction
            prediction = self._get_prediction_with_retry(conversation_messages, step.tool_call.message_index)

            # Add prediction to this step
            step.prediction = prediction

            # Track final goal prediction
            if prediction.prediction_type == "goal":
                final_prediction = prediction.content

            # Add model response to conversation
            response_content = f"[[{prediction.content}]]"
            conversation_messages.append({"role": "assistant", "content": response_content})

        # Update trajectory metadata
        trajectory.model_name = self.model_name
        trajectory.final_prediction = final_prediction

        return trajectory

    def _format_step_message(self, step: ToolCallStep, step_number: int) -> str:
        """Format a trajectory step for model input.

        Args:
            step: Tool call step to format
            step_number: Step number in sequence

        Returns:
            Formatted message string
        """
        tool_call = step.tool_call

        # Format database changes
        if step.database_changes:
            changes_str = ""
            for namespace, changes in step.database_changes.items():
                changes_str += f"{namespace}: +{len(changes)} records\\n"
            changes_str = changes_str.strip()
        else:
            changes_str = "No changes (read-only operation)"

        return self.task_prompt.format(
            step_number=step_number,
            tool_call=tool_call.python_function_string,
            tool_call_result=tool_call.result,
            changes_str=changes_str,
        )

    def _get_prediction_with_retry(self, messages: list[dict[str, str]], message_index: int) -> GoalInferencePrediction:
        """Get prediction from model with format error retry logic.

        Args:
            messages: Conversation messages so far
            message_index: sandbox_message_index for this step

        Returns:
            Parsed prediction
        """
        max_retries = 2

        for attempt in range(max_retries + 1):
            # Query model
            response: ModelResponse = self.llm.query(messages)
            model_output = response.choices[0].message.content or ""  # type: ignore

            try:
                # Parse response
                parsed = self.parser(model_output)
                return GoalInferencePrediction(
                    prediction_type=parsed["prediction_type"],
                    content=parsed["content"],
                    model_response=model_output,
                )
            except ValueError as e:
                if attempt < max_retries:
                    # Add error message and retry
                    error_msg = (
                        f"Format error: {e!s}\\n\\nYour previous response: {model_output}\\n\\nPlease fix the format."
                    )
                    messages.append({"role": "user", "content": error_msg})
                    continue
                else:
                    break

        # Final attempt failed, return error prediction
        return GoalInferencePrediction(
            prediction_type="error",
            content=f"Format error after {max_retries + 1} attempts",
            model_response=model_output,
        )
