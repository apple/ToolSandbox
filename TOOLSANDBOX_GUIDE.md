# ToolSandbox: Comprehensive Developer Guide

## 1. Project Overview

ToolSandbox is a stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. Unlike previous approaches that focus on stateless API evaluation or single-turn prompts, ToolSandbox introduces:

- **Stateful tool execution** with persistent world state
- **Implicit state dependencies** between tools requiring multi-step reasoning
- **Built-in user simulator** for on-policy conversational evaluation
- **Dynamic milestone-based evaluation** over arbitrary conversation trajectories

The project accompanies the [research paper](https://arxiv.org/abs/2408.04682) and provides insights into complex scenarios like State Dependency, Canonicalization, and Insufficient Information challenges.

## 2. Architecture & Core Concepts

### Execution Context

The [`ExecutionContext`](tool_sandbox/common/execution_context.py) is the central state management system storing:
- Complete tool sandbox state
- Dialog history between roles
- Database snapshots at every turn
- World state across multiple namespaces

```python
# Key databases managed by ExecutionContext
class DatabaseNamespace(StrEnum):
    SANDBOX = auto()      # Message history and tool traces
    SETTING = auto()      # Device settings (WiFi, cellular, etc.)
    CONTACT = auto()      # Contact book
    MESSAGING = auto()    # Text message history
    REMINDER = auto()     # Reminder database
```

### Role System

Four distinct roles interact via message passing (defined in [`base_role.py`](tool_sandbox/roles/base_role.py)):

- **SYSTEM**: Provides instructions and imports
- **USER**: Human or simulated user requesting task completion
- **AGENT**: LLM attempting to complete user tasks using tools
- **EXECUTION_ENVIRONMENT**: Executes tool calls and returns results

### Tool System

Tools are Python functions registered with the `@register_as_tool` decorator. See examples in [`contact.py`](tool_sandbox/tools/contact.py), [`messaging.py`](tool_sandbox/tools/messaging.py), etc.

```python
@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def remove_contact(person_id: str) -> None:
    """Remove an existing contact person to contact database."""
    current_context = get_current_context()
    current_context.remove_from_database(
        namespace=DatabaseNamespace.CONTACT,
        predicate=pl.col("person_id") == person_id
    )
```

### Scenario Framework

Scenarios extend base configurations defined in [`base_scenarios.py`](tool_sandbox/scenarios/base_scenarios.py). Each scenario includes:
- Starting context with initial database state
- Message sequences between roles
- Tool allow/deny lists
- Evaluation milestones

## 3. Installation & Setup

### Requirements
- Python 3.9+
- Virtual environment manager (conda recommended)

### Installation Steps
```bash
# Create environment
conda create -n ToolSandbox python=3.9
conda activate ToolSandbox

# Install dependencies
pip install '.[dev]'

# Setup pre-commit hooks (for development)
pre-commit install --hook-type pre-commit
```

### API Key Configuration

Set environment variables based on your chosen models (see [`cli/utils.py`](tool_sandbox/cli/utils.py) for all supported types):

```bash
# For OpenAI models (GPT-3.5, GPT-4, GPT-4o)
export OPENAI_API_KEY=<your_key>

# For Anthropic models (Claude)
export ANTHROPIC_API_KEY=<your_key>

# For search tools (optional)
export RAPID_API_KEY=<your_key>

# For Gemini models
export GOOGLE_CLOUD_PROJECT=<your_project>
export GOOGLE_CLOUD_REGION=<your_region>
```

## 4. Directory Structure Deep Dive

```
tool_sandbox/
├── cli/                    # Command-line interface
│   ├── __init__.py        # Main CLI entry point
│   └── utils.py           # Role type definitions and factories
├── common/                # Core framework components
│   ├── execution_context.py   # Central state management
│   ├── scenario.py            # Scenario definitions and results
│   ├── evaluation.py          # Milestone-based evaluation system
│   ├── tool_discovery.py      # Tool registration and discovery
│   └── message_conversion.py  # Message format conversions
├── roles/                 # Agent and user implementations
│   ├── base_role.py          # Abstract base for all roles
│   ├── anthropic_api_agent.py # Claude implementations
│   ├── openai_api_agent.py    # GPT implementations
│   ├── execution_environment.py # Tool execution handler
│   └── cli_role.py           # Human CLI interaction
├── tools/                 # Available tool implementations
│   ├── contact.py            # Contact management tools
│   ├── messaging.py          # SMS/messaging tools
│   ├── setting.py            # Device settings tools
│   ├── reminder.py           # Reminder management tools
│   └── utilities.py          # Utility functions
├── scenarios/             # Test scenario definitions
│   ├── base_scenarios.py         # Base scenario templates
│   ├── single_tool_call_scenarios.py  # Simple scenarios
│   └── multiple_tool_call_scenarios.py # Complex scenarios
├── analysis/              # Result analysis tools
│   ├── analysis.py           # Statistical analysis functions
│   └── data_loading.py       # Data loading utilities
└── notebooks/             # Jupyter analysis notebooks
```

## 5. Tool Development

### Creating New Tools

1. **Define the function** with proper type hints:
```python
from tool_sandbox.common.execution_context import get_current_context, DatabaseNamespace
from tool_sandbox.common.utils import register_as_tool, typechecked

@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def my_new_tool(param1: str, param2: int) -> dict[str, str]:
    """Google-style docstring describing the tool.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Returns:
        dict: Result description

    Raises:
        ValueError: When invalid input provided
    """
    # Tool implementation
    current_context = get_current_context()
    # Interact with databases as needed
    return {"result": "success"}
```

2. **Add to appropriate module** in [`tools/`](tool_sandbox/tools/) directory

3. **Database interactions** use Polars DataFrames:
```python
# Add to database
current_context.add_to_database(
    namespace=DatabaseNamespace.CONTACT,
    rows=[{"person_id": "123", "name": "John"}]
)

# Query database
results = current_context.get_database_snapshot(DatabaseNamespace.CONTACT)
filtered = results.filter(pl.col("name") == "John")
```

### Tool Visibility and Registration

Tools can be visible to specific roles using the `visible_to` parameter. The tool discovery system in [`tool_discovery.py`](tool_sandbox/common/tool_discovery.py) handles registration and backend selection.

## 6. Scenario Creation & Customization

### Extending Base Scenarios

Base scenarios are defined in [`base_scenarios.py`](tool_sandbox/scenarios/base_scenarios.py). Create extensions:

```python
from tool_sandbox.common.scenario import ScenarioExtension
from tool_sandbox.common.evaluation import Milestone, SnapshotConstraint

ScenarioExtension(
    name="my_custom_scenario",
    base_scenario=base_scenarios["base"],
    messages=[
        {
            "sender": RoleType.USER,
            "recipient": RoleType.AGENT,
            "content": "Turn off WiFi",
        }
    ],
    tool_allow_list=["set_wifi_status"],
    milestones=[
        Milestone(
            snapshot_constraints=[
                SnapshotConstraint(
                    database_namespace=DatabaseNamespace.SETTING,
                    snapshot_constraint=snapshot_similarity,
                    target_dataframe=pl.DataFrame({"wifi": False}),
                )
            ]
        )
    ],
)
```

### Scenario Categories

Scenarios are categorized for analysis (see [`execution_context.py`](tool_sandbox/common/execution_context.py)):
- **SINGLE_TOOL_CALL** / **MULTIPLE_TOOL_CALL**: Number of tools needed
- **SINGLE_USER_TURN** / **MULTIPLE_USER_TURN**: Conversation complexity
- **STATE_DEPENDENCY**: Requires specific world state setup
- **CANONICALIZATION**: Natural language → canonical form conversion
- **INSUFFICIENT_INFORMATION**: Missing tools/info (negative test)

### Tool Augmentations

Scenarios can include tool modifications:
- **DISTRACTION_TOOLS**: Add irrelevant tools
- **TOOL_NAME_SCRAMBLED**: Rename tools to generic names
- **ARG_DESCRIPTION_SCRAMBLED**: Remove argument descriptions
- **ARG_NAME_SCRAMBLED**: Use generic argument names

## 7. Role Implementation

### Available Role Types

See [`cli/utils.py`](tool_sandbox/cli/utils.py) for the complete list:

**Agents:**
- `Claude_3_Opus`, `Claude_3_Sonnet`, `Claude_3_Haiku`
- `GPT_3_5_0125`, `GPT_4_0125`, `GPT_4_o_2024_05_13`
- `Gemini_1_0`, `Gemini_1_5`, `Gemini_1_5_Flash`
- `Cohere_Command_R`, `Cohere_Command_R_Plus`
- `Gorilla`, `Hermes`, `MistralOpenAIServer`

**Users:**
- Same model types for user simulation
- `Cli` for human interaction

### Custom Agent Implementation

Extend [`BaseRole`](tool_sandbox/roles/base_role.py):

```python
class CustomAgent(BaseRole):
    def __init__(self):
        super().__init__(role_type=RoleType.AGENT)
        self.model_name = "custom-model"

    def generate_response(self, messages: list[Message]) -> str:
        # Implement your model's response generation
        return response
```

### Execution Environment

The [`ExecutionEnvironment`](tool_sandbox/roles/execution_environment.py) uses Python's `InteractiveConsole` to execute tool calls safely, capturing results and exceptions.

## 8. Evaluation System

### Milestone-Based Evaluation

Evaluation uses a DAG of milestones defined in [`evaluation.py`](tool_sandbox/common/evaluation.py). Each milestone specifies:

1. **Database constraints** with similarity measures
2. **Sequential dependencies** via DAG edges
3. **Column-wise matching** (exact, fuzzy, etc.)

### Similarity Measures

Available constraint types:
- **snapshot_similarity**: Direct database comparison
- **addition_similarity**: New rows added to reference
- **removal_similarity**: Rows removed from reference
- **update_similarity**: Rows modified in reference
- **guardrail_similarity**: Must match exactly (security check)

### Example Evaluation Definition

```python
milestones = [
    Milestone(  # Contact search
        snapshot_constraints=[
            SnapshotConstraint(
                database_namespace=DatabaseNamespace.SANDBOX,
                snapshot_constraint=snapshot_similarity,
                target_dataframe=pl.DataFrame({
                    "tool_trace": json.dumps({
                        "tool_name": "search_contacts",
                        "arguments": {"name": "Fredrik Thordendal"}
                    })
                })
            )
        ]
    ),
    Milestone(  # Message sent
        snapshot_constraints=[
            SnapshotConstraint(
                database_namespace=DatabaseNamespace.MESSAGING,
                snapshot_constraint=addition_similarity,
                target_dataframe=pl.DataFrame({
                    "recipient_phone_number": "+12453344098",
                    "content": "How's the new album coming along"
                })
            )
        ]
    )
]
edge_list = [(0, 1)]  # Contact search must happen before message
```

## 9. Running Experiments

### Single Scenario Execution

```bash
env ANTHROPIC_API_KEY=<key> OPENAI_API_KEY=<key> \
tool_sandbox --user GPT_4_o_2024_05_13 --agent Claude_3_Haiku --scenario wifi_off
```

### Batch Execution

```bash
# Run all scenarios
env OPENAI_API_KEY=<key> RAPID_API_KEY=<key> \
tool_sandbox --user GPT_4_o_2024_05_13 --agent GPT_4_o_2024_05_13

# Parallel execution
tool_sandbox --processes 4 --user <user_type> --agent <agent_type>
```

### Output Structure

Results are stored in `data/` with timestamped directories:
```
data/
└── agent_claude-3-haiku_user_gpt-4o_07_03_2024_00_17_44/
    ├── result_summary.json              # Aggregate results
    ├── scenario_results_polars.parquet  # Detailed results
    └── trajectories/
        └── wifi_off/
            └── conversation.json        # Full dialog
```

### Hosting Open Source Models

Use vLLM for open source models:
```bash
pip install vllm
env HF_TOKEN=<token> python3 -m vllm.entrypoints.openai.api_server \
--model gorilla-llm/gorilla-openfunctions-v2

# Then run scenarios
env OPENAI_API_KEY=EMPTY OPENAI_BASE_URL="http://0.0.0.0:8000/v1" \
tool_sandbox --agent Gorilla --scenario wifi_off
```

## 10. Analysis & Results

### Built-in Analysis

Use functions from [`analysis.py`](tool_sandbox/analysis/analysis.py):

```python
import polars as pl
from tool_sandbox.analysis.analysis import extract_meta_stats, extract_aggregated_stats

# Load results
df = pl.read_parquet("data/.../scenario_results_polars.parquet")

# Extract statistics
meta_stats = extract_meta_stats(df)
print(f"Success rate: {meta_stats['num_scenarios'][0] - meta_stats['num_exceptions'][0]} / {meta_stats['num_scenarios'][0]}")
```

### Jupyter Notebooks

Pre-built analysis notebooks in [`notebooks/`](tool_sandbox/notebooks/):
- `playground_single_run_analysis.ipynb`: Single experiment analysis
- `playground_agg_results_comparison.ipynb`: Multi-experiment comparison
- `trajectory_comparison.ipynb`: Conversation trajectory analysis

### Result Interpretation

Key metrics in `result_summary.json`:
- **similarity**: Overall milestone achievement score (0-1)
- **turn_count**: Number of conversation turns
- **milestone_mapping**: Which turns achieved which milestones
- **categories**: Scenario difficulty categories

## 11. Extending the Framework

### Adding New LLM Providers

1. Create agent class in [`roles/`](tool_sandbox/roles/) extending [`BaseRole`](tool_sandbox/roles/base_role.py)
2. Add to role factories in [`cli/utils.py`](tool_sandbox/cli/utils.py)
3. Update environment variable documentation

### Custom Similarity Measures

Implement `ColumnSimilarityMeasureType` protocol in [`evaluation.py`](tool_sandbox/common/evaluation.py):

```python
def custom_similarity(
    dataframe: pl.DataFrame,
    column_name: str,
    value: Any,
    atol_dict: Optional[dict[str, float]] = None,
) -> pl.DataFrame:
    # Return DataFrame with 'similarity' column (0-1 scores)
    pass
```

### New Tool Backends

Modify [`tool_discovery.py`](tool_sandbox/common/tool_discovery.py) to support additional tool formats beyond Python functions.

## 12. Development Guidelines

### Code Style

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Type hints required for all functions
- Use `pathlib.Path` instead of `os.path`
- Prefer native Python types over `typing` module
- Use f-strings for string formatting

### Testing

Run tests with pytest:
```bash
pytest tests/
# Parallel execution
pytest -n auto tests/
```

Test files follow the pattern `*_test.py` in [`tests/`](tests/) directory.

### Pre-commit Hooks

Configured hooks include:
- **ruff**: Code formatting and linting
- **mypy**: Type checking
- **pytest**: Test execution

Skip specific hooks if needed:
```bash
SKIP="mypy" git commit -m "temporary commit"
```

## 13. API Reference

### Core Classes

- **`ExecutionContext`**: Central state manager ([execution_context.py](tool_sandbox/common/execution_context.py))
- **`Scenario`**: Test case definition ([scenario.py](tool_sandbox/common/scenario.py))
- **`BaseRole`**: Abstract role interface ([base_role.py](tool_sandbox/roles/base_role.py))
- **`Milestone`**: Evaluation checkpoint ([evaluation.py](tool_sandbox/common/evaluation.py))

### Key Functions

- **`register_as_tool`**: Tool registration decorator
- **`get_current_context()`**: Access execution context
- **`named_scenarios()`**: Load all scenario definitions
- **`run_sandbox()`**: Main execution entry point

### Environment Variables

| Variable | Purpose | Required For |
|----------|---------|--------------|
| `OPENAI_API_KEY` | OpenAI API access | GPT models |
| `ANTHROPIC_API_KEY` | Anthropic API access | Claude models |
| `RAPID_API_KEY` | Search tools | Search scenarios |
| `GOOGLE_CLOUD_PROJECT` | GCP project | Gemini models |
| `HF_TOKEN` | Hugging Face access | Open source models |

## 14. Common Use Cases & Examples

### Research Evaluation

Compare model performance across categories:
```python
results = load_scenario_results()
state_dep_results = results.filter(
    pl.col("categories").list.contains("STATE_DEPENDENCY")
)
avg_similarity = state_dep_results["similarity"].mean()
```

### Custom Tool Development

Create domain-specific tools:
```python
@register_as_tool(visible_to=(RoleType.AGENT,))
def domain_specific_action(param: str) -> str:
    """Perform domain-specific action."""
    # Implementation
    return result
```

### Interactive Debugging

Use CLI mode for manual testing:
```bash
tool_sandbox --user Cli --agent Claude_3_Haiku --scenario custom_scenario
```

### Performance Optimization

For large-scale evaluation:
- Use `--processes N` for parallel execution
- Filter scenarios by category for targeted testing
- Cache model responses where possible

## 15. Appendices

### A. Scenario Categories Reference

- **SINGLE_TOOL_CALL**: Simple, one-tool tasks
- **MULTIPLE_TOOL_CALL**: Complex, multi-tool workflows
- **SINGLE_USER_TURN**: All info provided upfront
- **MULTIPLE_USER_TURN**: Requires clarification dialog
- **STATE_DEPENDENCY**: Must modify world state first
- **CANONICALIZATION**: Convert natural language to structured data
- **INSUFFICIENT_INFORMATION**: Missing required information
- **DISTRACTION_TOOLS**: Extra irrelevant tools present

### B. Tool Catalog

See individual tool files for complete API documentation:
- [**Contact Tools**](tool_sandbox/tools/contact.py): `add_contact`, `search_contacts`, `modify_contact`, `remove_contact`
- [**Messaging Tools**](tool_sandbox/tools/messaging.py): `send_message_with_phone_number`, `search_messages`
- [**Settings Tools**](tool_sandbox/tools/setting.py): `set_wifi_status`, `set_cellular_service_status`, `get_current_location`
- [**Reminder Tools**](tool_sandbox/tools/reminder.py): `add_reminder`, `search_reminder`, `modify_reminder`, `remove_reminder`
- [**Utility Tools**](tool_sandbox/tools/utilities.py): `get_current_timestamp`, `unit_conversion`, `datetime_info_to_timestamp`
- [**Search Tools**](tool_sandbox/tools/rapid_api_search_tools.py): `search_weather`, `search_stock`, `knowledge_base_question_answering`

### C. Troubleshooting

**Common Issues:**

1. **API Key Errors**: Verify environment variables are set correctly
2. **Import Errors**: Ensure virtual environment is activated and dependencies installed
3. **Permission Errors**: Check file permissions for output directory
4. **Model Timeout**: Increase timeout settings for slower models
5. **Memory Issues**: Reduce parallel processes or batch size

**Debug Mode:**
```bash
# Enable detailed logging
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Your test code here
"
```

For additional support, refer to the [research paper](https://arxiv.org/abs/2408.04682) or examine the test cases in [`tests/`](tests/) directory for usage examples.
