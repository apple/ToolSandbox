# Task Generator Implementation Plan

## Overview
This document outlines the implementation plan for adding a task generator utility to the Tool Sandbox codebase. The utility will generate realistic natural language task descriptions for mobile phone users based on available tools and current system state. This is a utility class for creating test scenarios, not a conversational role.

## 1. Core Architecture - Utility Class Approach

### 1.1 No Role Type Changes Needed
- TaskGenerator is **not a Role** - it doesn't participate in conversations
- No changes needed to `RoleType` enum
- No message-based communication required

### 1.2 CLI Integration (Optional)

**File:** `tool_sandbox/cli/utils.py`
**Changes:** Add task generator commands for scenario generation

```python
# Optional CLI commands for task generation
def generate_tasks_command():
    # pseudocode: CLI interface for generating tasks and creating scenarios
```

## 2. Core Implementation

### 2.1 Main Task Generator Utility

**File:** `tool_sandbox/common/task_generator.py`

#### 2.1.1 Data Structures

```python
@define
class GeneratedTask:
    """Container for a generated task."""
    task_id: str
    description: str
    required_tools: list[str]
    estimated_steps: Optional[int]
    tools_category: list[str]
    complexity: Optional[TaskComplexity]
    context_dependencies: Optional[list[str]]
```

#### 2.1.2 Tool Category System

```python
# Tool categories and their associated tools are discovered using existing functionality
# Uses get_tool_categories_info() from tool_discovery.py

# From tool_discovery.py:
# get_tool_categories_info(preferred_tool_backend: ToolBackend = ToolBackend.DEFAULT) -> Dict[str, ToolCategoryInfo]
# Returns: {"contact": ToolCategoryInfo(tools={...}, database=DatabaseNamespace.CONTACT), ...}

# TaskGenerator leverages this existing function instead of reimplementing discovery logic
# - Automatically discovers all tool categories from tool_sandbox.tools modules
# - Maps each category to its available tools and associated database
# - No static lists needed - always in sync with codebase
```

#### 2.1.3 Abstract Base Task Generator Class

```python
from abc import ABC, abstractmethod

class TaskGenerator(ABC):
    """Abstract base class for task generators."""

    def __init__(self) -> None:
        """Initialize task generator."""
        # pseudocode:
        # - load prompt templates from YAML file
        # - set up common configuration
        # - get all tool categories info

    def generate_task(
        self,
        tool_categories: list[str],
        execution_context: ExecutionContext,
        preferred_tool_backend: ToolBackend = ToolBackend.DEFAULT,
        max_retry_attempts: int = 3
    ) -> GeneratedTask:
        """Main entry point for task generation with feedback loop for format errors."""
        # pseudocode:
        # 1. validate_tool_categories(tool_categories)
        # 2. filter available tools by tool_categories
        # 3. get_tools_description()
        # 4. get_state_summary(execution_context)
        # 5. system_prompt, user_prompt = format_task_generation_prompt(tools, state, categories)
        #
        # 6. FEEDBACK LOOP with retry mechanism:
        # for attempt in range(max_retry_attempts):
        #     try:
        #         llm_response = model_inference(system_prompt, user_prompt)
        #         parsed_task = parse_and_validate_task(llm_response)
        #         return parsed_task
        #     except FormatError as e:
        #         if attempt < max_retry_attempts - 1:
        #             # For chat models, just send correction as new user message
        #             feedback_user_prompt = create_format_correction_prompt(
        #                 llm_response=llm_response,
        #                 error_details=str(e)
        #             )
        #             user_prompt = feedback_user_prompt  # Use feedback prompt for next attempt
        #         else:
        #             raise  # Re-raise after max attempts

    def create_format_correction_prompt(
        self,
        llm_response: str,
        error_details: str
    ) -> str:
        """Create a correction user prompt when LLM output format is incorrect."""
        # pseudocode:
        # 1. correction_template = load_format_correction_template()
        # 2. return format_template(
        #     incorrect_response=llm_response,
        #     error_explanation=error_details,
        #     expected_format=get_expected_format_example()
        # )

    def validate_tool_categories(self, tool_categories: list[str]) -> None:
        """Validate that requested tool categories exist in the codebase."""
        # pseudocode:
        # 1. available_categories = list(self.all_tool_categories_info.keys())
        # 2. invalid_categories = set(tool_categories) - set(available_categories)
        # 3. if invalid_categories: raise ValueError with available options

    def get_tools_description(self) -> str:
        """Get natural language description of available tools."""
        # pseudocode:
        # 1. return get_tool_docs_natural_language(self.available_tools)

    def get_state_summary(self, execution_context: ExecutionContext) -> str:
        """Get summary of execution context state."""
        # pseudocode:
        # 1. summary_parts = []
        # 2. for each database namespace relevant to tool categories:
        # 3.     if database has entries: summary_parts.append(summarize_database_content())
        # 4. return "\n".join(summary_parts) or "No relevant system state found."

    def format_task_generation_prompt(
        self,
        tools_description: str,
        state_summary: str,
        tool_categories: list[str]
    ) -> tuple[str, str]:
        """Format both system and user prompts for task generation.

        Args:
            tools_description: Description of available tools
            state_summary: Summary of execution context state
            tool_categories: List of tool categories to focus on

        Returns:
            tuple[str, str]: Tuple of (system_prompt, user_prompt)
        """
        # pseudocode:
        # 1. prompts_file = Path(__file__).parent / "task_generator_prompt.yaml"
        # 2. prompt_templates = yaml.safe_load(prompts_file.read_text())
        # 3. system_prompt = prompt_templates["system_prompt"]
        # 4. task_prompt_template = prompt_templates["task_prompt"]
        # 5. user_prompt = task_prompt_template.format(
        #     tools_description=tools_description,
        #     state_summary=state_summary,
        #     tool_categories=tool_categories
        # )
        # 6. return system_prompt, user_prompt

    @abstractmethod
    def model_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API with both system and user prompts. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def parse_and_validate_task(self, llm_response: str) -> GeneratedTask:
        """Parse LLM response and validate generated task. Must be implemented by subclasses."""
        # Each provider may have different parsing requirements based on their output format
        pass
```

#### 2.1.4 Provider-Specific Implementations

```python
class OpenAITaskGenerator(TaskGenerator):
    """OpenAI-specific task generator implementation."""

    def __init__(self, model_name: str = "gpt-4") -> None:
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI()

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def model_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API for task generation."""
        # pseudocode:
        # 1. messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": user_prompt}
        # ]
        # 2. response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     temperature=0.8,
        #     max_tokens=1000
        # )
        # 3. return response.choices[0].message.content

    def parse_and_validate_task(self, llm_response: str) -> GeneratedTask:
        """Parse OpenAI response and validate generated task."""
        # pseudocode:
        # 1. extract_task_components(llm_response)  # Extract TASK:, TRAJECTORY:, CATEGORIES:
        # 2. validate_required_fields(task_components)
        # 3. validate_tool_names_exist(trajectory_tools)
        # 4. return GeneratedTask(...)

class AnthropicTaskGenerator(TaskGenerator):
    """Anthropic-specific task generator implementation."""

    def __init__(self, model_name: str = "claude-3-sonnet-20240229") -> None:
        super().__init__()
        self.model_name = model_name
        self.client = anthropic.Anthropic()

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def model_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API for task generation."""
        # pseudocode:
        # 1. response = self.client.messages.create(
        #     model=self.model_name,
        #     max_tokens=1000,
        #     temperature=0.8,
        #     system=system_prompt,
        #     messages=[{"role": "user", "content": user_prompt}]
        # )
        # 2. return response.content[0].text

    def parse_and_validate_task(self, llm_response: str) -> GeneratedTask:
        """Parse Anthropic response and validate generated task."""
        # pseudocode:
        # 1. extract_task_components(llm_response)  # Extract TASK:, TRAJECTORY:, CATEGORIES:
        # 2. validate_required_fields(task_components)
        # 3. validate_tool_names_exist(trajectory_tools)
        # 4. return GeneratedTask(...)
```

## 3. Database Population Strategy (DEFERRED FOR MVP)

### 3.1 Current MVP Approach
**STATUS: SKIPPED FOR MINIMUM VIABLE IMPLEMENTATION**

For the initial implementation, we're focusing on getting the LLM integration and task generation working without sample data population. The `get_state_summary()` method will work with whatever data is already in the ExecutionContext.

### 3.2 Future Sample Data Implementation

**File:** `tool_sandbox/common/task_generator_sample_data.json` (FUTURE)

The sample data population system will be implemented in a future iteration to support:
- Automatic detection of empty databases
- Category-specific data population
- Consistent UUID generation for data relationships
- Realistic timestamp handling

### 3.3 MVP State Summary Approach

```python
def get_state_summary(self, execution_context: ExecutionContext) -> str:
    """Get summary of execution context state (MVP version)."""
    # pseudocode:
    # 1. summary_parts = []
    # 2. for each database namespace relevant to tool categories:
    # 3.     if database has entries: summary_parts.append(summarize_database_content())
    # 4. return "\n".join(summary_parts) or "No relevant system state found."
```

## 4. LLM Response Format and Feedback Loop

### 4.1 Expected Output Format

The LLM should generate tasks in this specific format:
```
TASK: [natural language task description]
TRAJECTORY: [tool_name_1 -> tool_name_2 -> tool_name_3]
CATEGORIES: [category1, category2, category3]
```

### 4.2 Feedback Loop Mechanism

When the LLM generates incorrectly formatted responses, the system implements a feedback loop:

1. **Format Detection**: `parse_and_validate_task()` detects format errors
2. **Error Classification**: Specific error types (missing fields, wrong format, invalid tools)
3. **Correction Prompt**: Generate feedback prompt explaining the error
4. **Retry with Feedback**: Use the correction prompt for the next attempt
5. **Max Attempts**: Fail after 3 attempts to prevent infinite loops

### 4.3 Format Correction Prompts

**File:** `tool_sandbox/scenarios/task_generation_prompts.yaml`

```yaml
format_correction: |
  Your previous response had formatting issues. Please fix the format and try again.

  YOUR PREVIOUS RESPONSE:
  {incorrect_response}

  ERROR DETAILS:
  {error_explanation}

  REQUIRED FORMAT:
  TASK: [clear, specific goal description]
  TRAJECTORY: [tool_name_1 -> tool_name_2 -> tool_name_3]
  CATEGORIES: [category1, category2, category3]

  Please generate a new response following the exact format above.

common_format_errors:
  missing_fields: "Missing required fields. Must include TASK:, TRAJECTORY:, and CATEGORIES: sections."
  invalid_trajectory_format: "TRAJECTORY must use format: tool_name_1 -> tool_name_2 -> tool_name_3"
  nonexistent_tools: "TRAJECTORY contains tools that don't exist in the available tool set: {invalid_tools}"
  category_mismatch: "CATEGORIES must match the requested tool categories: {expected_categories}"
```

### 4.4 Error Handling Classes

```python
class TaskGenerationError(Exception):
    """Base exception for task generation errors."""
    pass

class FormatError(TaskGenerationError):
    """Exception for LLM response format errors."""
    def __init__(self, message: str, response: str, expected_format: str):
        self.response = response
        self.expected_format = expected_format
        super().__init__(message)

class ValidationError(TaskGenerationError):
    """Exception for task validation errors."""
    pass
```

## 5. Usage Examples

### 5.1 Basic Usage

```python
from tool_sandbox.common.task_generator import OpenAITaskGenerator, AnthropicTaskGenerator

# Initialize specific task generator implementations
openai_gen = OpenAITaskGenerator(model_name="gpt-4")
anthropic_gen = AnthropicTaskGenerator(model_name="claude-3-sonnet-20240229")

# Get available tool categories using existing tool discovery
from tool_sandbox.common.tool_discovery import get_tool_categories_tools_map
tools_map = get_tool_categories_tools_map()
available_categories = list(tools_map.keys())
# Returns: ["contact", "messaging", "reminder", "utilities", "setting", "rapid_api_search_tools"]

# Generate task for specific tool categories
context = ExecutionContext()  # Empty context - will be automatically populated
communication_task = openai_gen.generate_task(
    tool_categories=["contact", "messaging"],
    execution_context=context  # Gets populated with sample contacts & messages
)
# Returns: GeneratedTask(description="Reply to Sarah's message about lunch plans", ...)

# Generate task for reminder category
reminder_task = anthropic_gen.generate_task(
    tool_categories=["reminder"],
    execution_context=ExecutionContext()  # Gets populated with sample reminders
)
# Returns: GeneratedTask(description="Check your overdue reminder about Company SF tickets", ...)

# Generate task for multiple categories
mixed_task = openai_gen.generate_task(
    tool_categories=["messaging", "reminder", "utilities"],
    execution_context=ExecutionContext()  # Gets populated with relevant sample data
)

# Generate task with pre-existing context
existing_context = ExecutionContext()
# ... context already has some data ...
info_task = openai_gen.generate_task(
    tool_categories=["rapid_api_search_tools", "setting"],
    execution_context=existing_context  # Uses existing data, no sample population needed
)
```

### 5.2 Integration with Scenario Creation

```python
def create_scenario_from_generated_task(task: GeneratedTask, execution_context: ExecutionContext) -> Scenario:
    """Convert generated task into Tool Sandbox scenario."""
    # pseudocode:
    # 1. scenario = Scenario()
    # 2. scenario.starting_context = deepcopy(execution_context)  # Use populated context
    # 3. scenario.starting_context.add_to_database(
    #      namespace=DatabaseNamespace.SANDBOX,
    #      rows=[{"sender": RoleType.USER, "recipient": RoleType.AGENT, "content": task.description}]
    #    )
    # 4. scenario.evaluation = create_evaluation_from_task(task)
    # 5. return scenario
```

## 6. Testing Strategy

### 6.1 Unit Tests

**File:** `tests/common/task_generator_test.py`

```python
class TestTaskGenerator:
    def test_get_tools_description(self):
        # Test tool filtering and description generation
        pass

    def test_get_state_summary(self):
        # Test state extraction from execution context
        pass

    def test_generate_task(self):
        # Test end-to-end single task generation
        pass

    def test_parse_and_validate_task(self):
        # Test parsing of various LLM response formats
        pass

    def test_populate_sample_data(self):
        # Test sample data population for different categories
        pass

    def test_context_needs_population(self):
        # Test detection of when sample data is needed
        pass

class TestProviderSpecificGenerators:
    def test_openai_generator(self):
        # Test OpenAI-specific implementation
        pass

    def test_anthropic_generator(self):
        # Test Anthropic-specific implementation
        pass
```

### 6.2 Integration Tests

```python
class TestTaskGeneratorIntegration:
    def test_scenario_creation_from_task(self):
        # Test integration with scenario creation from single task
        pass

    def test_with_real_execution_context(self):
        # Test with actual execution context states
        pass

    def test_tool_integration(self):
        # Test with real tool discovery and conversion
        pass
```

## 7. Error Handling and Edge Cases

### 7.1 LLM Response Handling

```python
def robust_json_extraction(llm_response: str) -> Dict[str, Any]:
    """Extract JSON from various LLM response formats."""
    # pseudocode:
    # 1. try direct json.loads(llm_response)
    # 2. try extracting JSON block with ```json markers
    # 3. try regex extraction of JSON structure
    # 4. fallback to manual parsing of structured text
    # 5. raise ParseError if all methods fail
```

### 7.2 Validation and Fallbacks

```python
def validate_and_fix_task(task: Dict[str, Any], available_tools: List[str]) -> GeneratedTask:
    """Validate task and attempt to fix common issues."""
    # pseudocode:
    # - check if required_tools exist in available_tools
    # - estimate steps if missing
    # - fix common formatting issues
    # - raise ValidationError if task can't be fixed
```

## 8. Future Enhancements

### 8.1 Advanced Features (Post-MVP)

1. **Randomized Sample Data**: Randomly select from larger pool of sample data entries
2. **Difficulty Progression**: Generate tasks of varying complexity levels
3. **User Persona Integration**: Generate tasks based on user profiles (student, professional, etc.)
4. **Temporal Awareness**: Generate time-sensitive tasks based on current time/date
5. **State Consistency**: Ensure generated tasks are consistent with current system state
6. **Constraint Satisfaction**: Ensure generated tasks can actually be completed with available tools

### 8.2 Performance Optimizations

1. **Template Caching**: Cache formatted prompt templates
2. **Batch Generation**: Generate multiple task sets in parallel
3. **Smart Retries**: Implement exponential backoff with jitter
4. **Response Caching**: Cache LLM responses for similar contexts

## 9. Dependencies and Requirements

### 9.1 New Dependencies
- No new external dependencies required (reuse existing OpenAI, Anthropic clients)

### 9.2 Internal Dependencies
- `tool_sandbox.common.tool_discovery.get_tool_categories_tools_map` (for tool category/tools mapping)
- `tool_sandbox.common.tool_conversion.get_tool_docs_natural_language`
- `tool_sandbox.common.execution_context.ExecutionContext` (for system state and tool filtering)
- `tool_sandbox.common.utils.deterministic_uuid` (for consistent sample data IDs)
- Existing retry mechanisms from agent implementations
- YAML prompt loading pattern from Hermes agent

## 10. File Structure Summary

```
tool_sandbox/
├── common/
│   ├── task_generator.py               # Main utility class (NEW)
│   ├── task_generator_sample_data.json # Sample data for population (NEW)
│   └── task_generator_prompts.yaml    # Prompt templates (NEW)
├── cli/
│   └── utils.py                        # Optional CLI commands (OPTIONAL)
└── tests/
    └── common/
        └── task_generator_test.py      # Unit tests (NEW)
```

This implementation plan provides a utility-based approach for task generation that integrates with the existing Tool Sandbox architecture without polluting the conversational role system.
