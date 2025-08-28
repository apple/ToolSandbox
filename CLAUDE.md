# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository for goal inference research in LLM tool use environments.

## Project Overview

ToolSandbox is a stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. **We are using it as the foundation for research into goal inference** - investigating whether LLMs can accurately infer user goals by observing tool call sequences alone.

### Research Context
This codebase supports a 3-component research pipeline:
1. **Goal Inference Task Generator**: Creates realistic tasks optimized for goal inference research
2. **Trajectory Generator**: Executes tasks using ToolSandbox agents to collect tool call sequences
3. **Goal Inference Agent**: Analyzes tool trajectories to predict original user goals

## Common Development Commands

### Installation and Setup
```bash
# Create virtual environment
conda create -n ToolSandbox python=3.9
conda activate ToolSandbox

# Install dependencies (development mode)
pip install '.[dev]'

# Setup pre-commit hooks
pre-commit install --hook-type pre-commit
```

### Testing and Code Quality
```bash
# Run all tests with parallel execution
pytest -n auto

# Run specific test file
pytest tests/common/evaluation_test.py

# Run linting and formatting
ruff check .
ruff format .

# Type checking
mypy tool_sandbox

# Skip specific pre-commit hook if needed
SKIP="mypy" git commit -m "commit message"
```

### Running ToolSandbox for Research
```bash
# Single scenario for trajectory collection
tool_sandbox --user GPT_4_o_2024_05_13 --agent Claude_3_Haiku --scenarios wifi_off

# Run all GOAL_INFERENCE scenarios (by category)
tool_sandbox --user GPT_4_o_2024_05_13 --agent Claude_3_Haiku --scenarios GOAL_INFERENCE

# Run specific goal inference scenarios (by individual names)
tool_sandbox --user GPT_4_o_2024_05_13 --agent Claude_3_Haiku --scenarios dinner_plan_review_and_confirmation coordinate_ride_with_distance

# Batch trajectory generation for research dataset (all scenarios)
tool_sandbox --user GPT_4_o_2024_05_13 --agent GPT_4_o_2024_05_13

# Interactive mode for task design validation
tool_sandbox --user Cli --agent Claude_3_Haiku --scenarios custom_scenario
```

## Architecture Overview

### Core Components
- **ExecutionContext** (`tool_sandbox/common/execution_context.py`): Central state management storing world state across databases (SANDBOX, SETTING, CONTACT, MESSAGING, REMINDER)
- **Role System** (`tool_sandbox/roles/`): Four roles interact via message passing - SYSTEM, USER, AGENT, EXECUTION_ENVIRONMENT
- **Tool System** (`tool_sandbox/tools/`): Python functions registered with `@register_as_tool` decorator
- **Scenario Framework** (`tool_sandbox/scenarios/`): Extends base scenarios with specific tasks, tool lists, and evaluation milestones
- **Evaluation System** (`tool_sandbox/common/evaluation.py`): Milestone-based evaluation using DAG of checkpoints with similarity measures

### Research-Specific Extensions
When working on goal inference research, focus on these components:
- **Task Generation**: Create scenarios that force information lookup and cross-referencing
- **Trajectory Collection**: Capture detailed tool call sequences with parameters
- **Goal Inference Analysis**: Extract and analyze patterns from collected trajectories

### Key File Structure
```
tool_sandbox/
├── cli/                    # Command-line interface
├── common/                 # Core framework (execution context, evaluation, scenarios)
├── roles/                  # Agent implementations (Claude, GPT, Gemini, etc.)
├── tools/                  # Available tools (contact, messaging, settings, etc.)
├── scenarios/              # Test scenario definitions
└── analysis/               # Result analysis utilities (extend for goal inference)
```

## Tool Development

### Creating New Tools
Tools are Python functions with the `@register_as_tool` decorator:

```python
from tool_sandbox.common.execution_context import get_current_context, DatabaseNamespace
from tool_sandbox.common.utils import register_as_tool, typechecked

@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def my_tool(param1: str, param2: int) -> dict[str, str]:
    """Google-style docstring describing the tool.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Returns:
        dict: Result description

    Raises:
        ValueError: When invalid input provided
    """
    current_context = get_current_context()
    # Database interactions use Polars DataFrames
    current_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[{"person_id": "123", "name": "John"}]
    )
    return {"result": "success"}
```

### Research-Relevant Tool Categories
Current tools support these goal inference patterns:
- **Communication**: `search_messages`, `send_message_with_phone_number`, `search_contacts`
- **Location/Weather**: `get_current_location`, `search_weather_around_lat_lon`
- **Time Management**: `search_reminder`, `add_reminder`, `get_current_timestamp`
- **System Status**: `get_wifi_status`, `get_cellular_service_status`
- **Financial**: `search_stock`, `convert_currency`

### Database Operations
- Use `get_current_context()` to access execution context
- Databases are Polars DataFrames stored in different namespaces
- Common operations: `add_to_database()`, `get_database_snapshot()`, `remove_from_database()`

## Scenario Creation

Scenarios extend base configurations in `tool_sandbox/scenarios/base_scenarios.py`. Key elements:
- Starting context with initial database state
- Message sequences between roles
- Tool allow/deny lists for filtering available tools
- Evaluation milestones defining success criteria

### Scenario Categories
- **SINGLE_TOOL_CALL/MULTIPLE_TOOL_CALL**: Tool complexity
- **SINGLE_USER_TURN/MULTIPLE_USER_TURN**: Conversation complexity
- **STATE_DEPENDENCY**: Requires world state modifications
- **CANONICALIZATION**: Natural language → structured data conversion
- **INSUFFICIENT_INFORMATION**: Missing required information (negative tests)
- **GOAL_INFERENCE**: Scenarios designed for goal inference research

## Research Data Collection

### Trajectory Collection Pipeline
1. **Task Generation**: Use research-optimized scenarios
2. **Agent Execution**: Run multiple models on same tasks
3. **Trajectory Extraction**: Capture tool sequences with parameters
4. **Goal Inference Testing**: Analyze patterns for goal prediction

### Research-Focused Scenario Categories
Focus on scenarios that create good goal inference opportunities:
- **GOAL_INFERENCE**: Dedicated scenarios for goal inference research
- **MULTIPLE_TOOL_CALL**: Complex sequences required
- **STATE_DEPENDENCY**: World state affects tool effectiveness
- **INFORMATION_LOOKUP**: Cross-referencing multiple data sources
- **PROGRESSIVE_DISAMBIGUATION**: Goals become clearer over time

## Environment Variables

| Variable | Purpose | Models |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API access | GPT-3.5, GPT-4, GPT-4o |
| `ANTHROPIC_API_KEY` | Anthropic API access | Claude models |
| `RAPID_API_KEY` | Search tools | Search scenarios (optional) |
| `GOOGLE_CLOUD_PROJECT` | GCP project | Gemini models |
| `GOOGLE_CLOUD_REGION` | GCP region | Gemini models |
| `HF_TOKEN` | Hugging Face access | Open source models |

## Code Style Requirements

- Use Google-style docstrings for all functions
- Type hints required (Python 3.9+ style preferred over `typing` module)
- Use `pathlib.Path` instead of `os.path`
- Use f-strings for string formatting
- Follow existing patterns in tool registration and database interactions
- Tools must be defensive against invalid inputs and raise appropriate exceptions
- When writing new files, do not add copyright notices.
- When editing existing files, do not remove copyright notices.

### Linting and Formatting Guidelines
- **No trailing whitespace** - Remove any spaces at the end of lines
- **No whitespace in blank lines** - Blank lines should be completely empty
- **Use set comprehensions** - Use `{str(x) for x in items}` instead of `set(str(x) for x in items)`
- **Avoid unnecessary list() calls** - Use `sorted(items)` instead of `sorted(list(items))`
- **Follow existing noqa patterns** - When suppressing linting warnings, follow existing project patterns

### Research-Specific Code Patterns
When extending for goal inference research:
- **Trajectory Logging**: Capture full tool call sequences with timestamps and parameters
- **Goal Annotation**: Include ground truth goals for evaluation
- **Progressive Analysis**: Support partial trajectory evaluation
- **Data Export**: Ensure trajectory data can be exported for analysis

## Output and Results

### Standard ToolSandbox Output
Results are stored in `data/` with timestamped directories:
- `result_summary.json`: Aggregate evaluation results
- `scenario_results_polars.parquet`: Detailed results for analysis
- `trajectories/`: Full conversation logs per scenario

### Research-Specific Analysis
For goal inference research, extend analysis to include:
- **Tool Sequence Patterns**: Common patterns for different goal types
- **Goal Inference Accuracy**: Model performance at predicting goals from partial trajectories
- **Ambiguity Analysis**: How uncertainty changes over tool call sequences
- **Cross-Model Comparison**: Goal inference capabilities across different LLMs

Use analysis notebooks in `notebooks/` for result interpretation and comparison. Create research-specific notebooks for goal inference pattern analysis.

## Technical Implementation Notes

### Working with Trajectories
```python
# Extract tool call sequences from trajectory data
def extract_tool_sequence(trajectory_data):
    tool_calls = []
    for message in trajectory_data['messages']:
        if message['role'] == 'AGENT' and 'tool_calls' in message:
            for tool_call in message['tool_calls']:
                tool_calls.append({
                    'tool_name': tool_call['name'],
                    'parameters': tool_call['parameters'],
                    'timestamp': message['timestamp'],
                    'result': tool_call.get('result')
                })
    return tool_calls
```

### Database Schema Access
```python
# Access current database state for analysis
current_context = get_current_context()
contacts_df = current_context.get_database_snapshot(DatabaseNamespace.CONTACT)
messages_df = current_context.get_database_snapshot(DatabaseNamespace.MESSAGING)
reminders_df = current_context.get_database_snapshot(DatabaseNamespace.REMINDER)
```

### Custom Evaluation Metrics
```python
# Extend evaluation for goal inference metrics
from tool_sandbox.common.evaluation import BaseEvaluator

class GoalInferenceEvaluator(BaseEvaluator):
    def evaluate_trajectory(self, trajectory, ground_truth_goal):
        # Custom evaluation logic for goal inference
        pass
```

This codebase serves dual purposes: maintaining ToolSandbox as a general tool use benchmark AND supporting our specific research into goal inference from tool usage patterns.
- Remember to run `ruff check` on modified Python files after implementation. Add new linting patterns to CLAUDE.md Linting and Formatting Guidelines section. Don't attempt to fix the linting issues.
- When asked to create a git commit message, give me the commit message instead of generating a command directly. Follow previous commit message styling. Don't include more than one line about changes to any md files.
- When generating commit message, don't include generated with or co-authored by claude code.
