# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToolSandbox is a stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. It provides a framework for testing agents with persistent world state, implicit tool dependencies, and milestone-based evaluation over conversation trajectories.

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

### Running ToolSandbox
```bash
# Single scenario with specific models
env ANTHROPIC_API_KEY=<key> OPENAI_API_KEY=<key> \
tool_sandbox --user GPT_4_o_2024_05_13 --agent Claude_3_Haiku --scenario wifi_off

# Run all scenarios
env OPENAI_API_KEY=<key> RAPID_API_KEY=<key> \
tool_sandbox --user GPT_4_o_2024_05_13 --agent GPT_4_o_2024_05_13

# Interactive CLI mode for debugging
tool_sandbox --user Cli --agent Claude_3_Haiku --scenario custom_scenario
```

## Architecture Overview

### Core Components
- **ExecutionContext** (`tool_sandbox/common/execution_context.py`): Central state management storing world state across databases (SANDBOX, SETTING, CONTACT, MESSAGING, REMINDER)
- **Role System** (`tool_sandbox/roles/`): Four roles interact via message passing - SYSTEM, USER, AGENT, EXECUTION_ENVIRONMENT
- **Tool System** (`tool_sandbox/tools/`): Python functions registered with `@register_as_tool` decorator
- **Scenario Framework** (`tool_sandbox/scenarios/`): Extends base scenarios with specific tasks, tool lists, and evaluation milestones
- **Evaluation System** (`tool_sandbox/common/evaluation.py`): Milestone-based evaluation using DAG of checkpoints with similarity measures

### Key File Structure
```
tool_sandbox/
├── cli/                    # Command-line interface
├── common/                 # Core framework (execution context, evaluation, scenarios)
├── roles/                  # Agent implementations (Claude, GPT, Gemini, etc.)
├── tools/                  # Available tools (contact, messaging, settings, etc.)
├── scenarios/              # Test scenario definitions
└── analysis/               # Result analysis utilities
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

## Output and Results

Results are stored in `data/` with timestamped directories:
- `result_summary.json`: Aggregate evaluation results
- `scenario_results_polars.parquet`: Detailed results for analysis
- `trajectories/`: Full conversation logs per scenario

Use analysis notebooks in `notebooks/` for result interpretation and comparison.
