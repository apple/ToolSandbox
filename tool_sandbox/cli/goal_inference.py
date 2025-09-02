"""CLI for running goal inference experiments."""

import argparse
import json

from tool_sandbox.models.goal_inference_agent import GoalInferenceAgent


def main() -> None:
    """Main CLI entry point for goal inference experiments."""
    parser = argparse.ArgumentParser(description="Run goal inference on ToolSandbox trajectories")
    parser.add_argument("trajectory_path", help="Path to trajectory directory")
    parser.add_argument("--model", required=True, help="LiteLLM model name (e.g., gpt-4o, claude-3-sonnet-20240229)")
    parser.add_argument("--output", help="Output file for results (JSON format)")

    args = parser.parse_args()

    # Initialize agent
    agent = GoalInferenceAgent(model_name=args.model)

    # Run goal inference
    trajectory = agent.run_goal_inference_on_trajectory_path(args.trajectory_path)

    # Print results
    print(f"Trajectory: {trajectory.scenario_name}")
    print(f"Model: {trajectory.model_name}")
    print(f"Total Steps: {len(trajectory.steps)}")
    print(f"Final Prediction: {trajectory.final_prediction}")
    print("\\nStep-by-step analysis:")

    for i, step in enumerate(trajectory.steps):
        tool_call = step.tool_call
        prediction = step.prediction
        print(f"\\n  Step {i + 1} (message_index: {tool_call.message_index}):")
        print(f"    Tool: {tool_call.python_function_string}")
        print(f"    Result: {tool_call.result}")
        if step.database_changes:
            changes_str = ", ".join([f"{ns}: +{len(changes)}" for ns, changes in step.database_changes.items()])
            print(f"    DB Changes: {changes_str}")
        else:
            print("    DB Changes: No changes (read-only)")
        if prediction:
            print(f"    Prediction: [{prediction.prediction_type}] {prediction.content}")
        else:
            print("    Prediction: No prediction")

    # Save to file if requested
    if args.output:
        output_data = {
            "trajectory_name": trajectory.scenario_name,
            "model_name": trajectory.model_name,
            "total_steps": len(trajectory.steps),
            "final_prediction": trajectory.final_prediction,
            "steps": [
                {
                    "message_index": step.tool_call.message_index,
                    "tool_call": {
                        "tool_name": step.tool_call.tool_name,
                        "python_function_string": step.tool_call.python_function_string,
                        "arguments": step.tool_call.arguments,
                        "result": step.tool_call.result,
                    },
                    "database_changes": step.database_changes,
                    "prediction": {
                        "prediction_type": step.prediction.prediction_type,
                        "content": step.prediction.content,
                        "model_response": step.prediction.model_response,
                    }
                    if step.prediction
                    else None,
                }
                for step in trajectory.steps
            ],
            "initial_database_state": trajectory.initial_database_state,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
