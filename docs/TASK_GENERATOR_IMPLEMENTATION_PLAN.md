# Task Generator Role Implementation Plan

## Overview
This document outlines the implementation plan for adding a task generator role to the Tool Sandbox codebase. The role will generate realistic natural language task descriptions for mobile phone users based on available tools and current system state.

## 1. Core Architecture Changes

### 1.1 Execution Context Updates

**File:** `tool_sandbox/common/execution_context.py`
**Changes:** Add new role type to enum

```python
class RoleType(StrEnum):
    # ... existing roles ...
    TASK_GENERATOR = auto()  # Add this line around line 55
```

### 1.2 CLI Integration (Optional)

**File:** `tool_sandbox/cli/utils.py`
**Changes:** Add task generator to factory mappings

```python
class RoleImplType(StrEnum):
    # ... existing types ...
    Task_Generator_GPT4 = auto()
    Task_Generator_Claude = auto()

TASK_GENERATOR_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.Task_Generator_GPT4: lambda: OpenAITaskGeneratorRole(model_name="gpt-4"),
    RoleImplType.Task_Generator_Claude: lambda: AnthropicTaskGeneratorRole(model_name="claude-3-sonnet"),
}
```

## 2. Core Implementation

### 2.1 Main Task Generator Role

**File:** `tool_sandbox/roles/task_generator_role.py`

#### 2.1.1 Data Structures

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import auto
from strenum import StrEnum

class TaskCategory(StrEnum):
    COMMUNICATION = auto()
    SCHEDULING = auto()
    INFORMATION_LOOKUP = auto()
    LOCATION_BASED = auto()
    MULTI_STEP = auto()
    SOCIAL = auto()

class TaskComplexity(StrEnum):
    SIMPLE = auto()      # 1-2 tool calls
    MODERATE = auto()    # 3-5 tool calls
    COMPLEX = auto()     # 6+ tool calls

@dataclass
class GeneratedTask:
    """Container for a generated task."""
    description: str
    required_tools: List[str]
    estimated_steps: int
    category: TaskCategory
    complexity: TaskComplexity
    context_dependencies: List[str]  # e.g., ["has_contacts", "location_enabled"]
```

#### 2.1.2 Base Task Generator Class

```python
class TaskGeneratorRole(BaseRole):
    """Base class for task generator roles."""

    role_type: RoleType = RoleType.TASK_GENERATOR

    def __init__(self, model_name: str) -> None:
        """Initialize task generator."""
        # pseudocode:
        # - call super().__init__()
        # - store model_name
        # - initialize task generation templates
        # - set up tool filtering capabilities

    def generate_tasks(
        self,
        num_tasks: int = 5,
        tool_filter: Optional[List[str]] = None,
        complexity: Optional[TaskComplexity] = None,
        category: Optional[TaskCategory] = None,
        use_current_state: bool = True
    ) -> List[GeneratedTask]:
        """Main entry point for task generation."""
        # pseudocode:
        # 1. get_available_tools_description(tool_filter)
        # 2. get_system_state_summary() if use_current_state
        # 3. create_task_generation_prompt(tools, state, constraints)
        # 4. call_llm_for_tasks(prompt)
        # 5. parse_and_validate_tasks(llm_response)
        # 6. return List[GeneratedTask]

    def get_available_tools_description(self, tool_filter: Optional[List[str]] = None) -> str:
        """Get natural language description of available tools."""
        # pseudocode:
        # 1. context = get_current_context()
        # 2. all_tools = context.get_available_tools(scrambling_allowed=False)
        # 3. if tool_filter: filter tools by tool_filter
        # 4. return get_tool_docs_natural_language(filtered_tools)

    def get_system_state_summary(self) -> Dict[str, Any]:
        """Extract relevant system state information."""
        # pseudocode:
        # 1. context = get_current_context()
        # 2. contacts_summary = self._summarize_contacts_db(context)
        # 3. messages_summary = self._summarize_messages_db(context)
        # 4. reminders_summary = self._summarize_reminders_db(context)
        # 5. settings_summary = self._summarize_settings_db(context)
        # 6. return combined summary dict

    def _summarize_contacts_db(self, context: ExecutionContext) -> Dict[str, Any]:
        """Summarize contacts database state."""
        # pseudocode:
        # 1. contacts_df = context.get_database(DatabaseNamespace.CONTACT)
        # 2. return {
        #     "total_contacts": len(contacts_df),
        #     "has_self_contact": any(contacts_df["is_self"]),
        #     "relationship_types": list(contacts_df["relationship"].unique()),
        #     "sample_names": contacts_df["name"].head(3).to_list()
        # }

    def _summarize_messages_db(self, context: ExecutionContext) -> Dict[str, Any]:
        """Summarize messaging database state."""
        # pseudocode:
        # 1. messages_df = context.get_database(DatabaseNamespace.MESSAGING)
        # 2. return {
        #     "total_messages": len(messages_df),
        #     "recent_conversations": get_recent_conversation_summary(),
        #     "unique_contacts_messaged": get_unique_message_contacts()
        # }

    def _summarize_reminders_db(self, context: ExecutionContext) -> Dict[str, Any]:
        """Summarize reminders database state."""
        # pseudocode:
        # 1. reminders_df = context.get_database(DatabaseNamespace.REMINDER)
        # 2. current_time = get_current_timestamp()
        # 3. return {
        #     "total_reminders": len(reminders_df),
        #     "upcoming_reminders": count_upcoming_reminders(current_time),
        #     "location_based_reminders": count_location_reminders()
        # }

    def _summarize_settings_db(self, context: ExecutionContext) -> Dict[str, Any]:
        """Summarize system settings state."""
        # pseudocode:
        # 1. settings_df = context.get_database(DatabaseNamespace.SETTING)
        # 2. return {
        #     "wifi_enabled": get_setting_value("wifi_status"),
        #     "cellular_enabled": get_setting_value("cellular_service_status"),
        #     "location_enabled": get_setting_value("location_service_status"),
        #     "low_battery_mode": get_setting_value("low_battery_mode_status")
        # }

    def create_task_generation_prompt(
        self,
        tools_description: str,
        state_summary: Dict[str, Any],
        num_tasks: int,
        constraints: Dict[str, Any]
    ) -> str:
        """Create the prompt for LLM task generation."""
        # pseudocode:
        # 1. base_prompt = load_base_task_generation_template()
        # 2. context_section = format_context_information(state_summary)
        # 3. tools_section = format_tools_information(tools_description)
        # 4. constraints_section = format_constraints(constraints)
        # 5. examples_section = load_task_examples()
        # 6. output_format = specify_json_output_format()
        # 7. return combine_prompt_sections(...)

    def call_llm_for_tasks(self, prompt: str) -> str:
        """Abstract method for LLM API calls."""
        raise NotImplementedError("Subclasses must implement this method")

    def parse_and_validate_tasks(self, llm_response: str) -> List[GeneratedTask]:
        """Parse LLM response and validate generated tasks."""
        # pseudocode:
        # 1. try: parsed_json = json.loads(extract_json_from_response(llm_response))
        # 2. tasks = []
        # 3. for task_data in parsed_json["tasks"]:
        #     - validate_task_structure(task_data)
        #     - validate_required_tools_exist(task_data["required_tools"])
        #     - task = GeneratedTask(**task_data)
        #     - tasks.append(task)
        # 4. return tasks

    def validate_task_structure(self, task_data: Dict[str, Any]) -> bool:
        """Validate that task has required fields."""
        # pseudocode:
        # required_fields = ["description", "required_tools", "estimated_steps", "category"]
        # return all(field in task_data for field in required_fields)

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Required by BaseRole - not used for task generation."""
        pass
```

#### 2.1.3 OpenAI Task Generator Implementation

```python
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

class OpenAITaskGeneratorRole(TaskGeneratorRole):
    """OpenAI-specific task generator implementation."""

    def __init__(self, model_name: str = "gpt-4") -> None:
        super().__init__(model_name)
        self.client = OpenAI()
        self.model_name = model_name

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def call_llm_for_tasks(self, prompt: str) -> str:
        """Call OpenAI API for task generation."""
        # pseudocode:
        # 1. messages = [{"role": "user", "content": prompt}]
        # 2. response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=messages,
        #     temperature=0.8,  # Higher for creativity
        #     max_tokens=2000
        # )
        # 3. return response.choices[0].message.content
```

#### 2.1.4 Anthropic Task Generator Implementation

```python
import anthropic
from tenacity import retry, wait_random_exponential, stop_after_attempt

class AnthropicTaskGeneratorRole(TaskGeneratorRole):
    """Anthropic-specific task generator implementation."""

    def __init__(self, model_name: str = "claude-3-sonnet-20240229") -> None:
        super().__init__(model_name)
        self.client = anthropic.Anthropic()
        self.model_name = model_name

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def call_llm_for_tasks(self, prompt: str) -> str:
        """Call Anthropic API for task generation."""
        # pseudocode:
        # 1. response = self.client.messages.create(
        #     model=self.model_name,
        #     max_tokens=2000,
        #     temperature=0.8,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # 2. return response.content[0].text
```

## 3. Prompt Engineering Strategy

### 3.1 Base Prompt Template

```python
TASK_GENERATION_TEMPLATE = """
You are a task generator for a mobile phone assistant. Generate realistic, natural tasks that a typical smartphone user might want to accomplish.

AVAILABLE TOOLS:
{tools_description}

CURRENT SYSTEM STATE:
{state_summary}

REQUIREMENTS:
- Generate {num_tasks} unique tasks
- Tasks should be realistic and natural
- Use the provided tools and current system state as context
- Vary complexity from simple (1-2 steps) to complex (5+ steps)
- Include diverse scenarios: communication, scheduling, information lookup, etc.

TASK CATEGORIES:
- COMMUNICATION: messaging, calling, contact management
- SCHEDULING: reminders, appointments, time management
- INFORMATION_LOOKUP: weather, location, search, currency
- LOCATION_BASED: navigation, nearby places, location reminders
- MULTI_STEP: tasks requiring multiple tool calls and coordination
- SOCIAL: group activities, event planning, social coordination

OUTPUT FORMAT (JSON):
{{
  "tasks": [
    {{
      "description": "Natural language task description",
      "required_tools": ["tool1", "tool2"],
      "estimated_steps": 3,
      "category": "COMMUNICATION",
      "complexity": "MODERATE",
      "context_dependencies": ["has_contacts"]
    }}
  ]
}}

EXAMPLES:
Good: "Send a message to Sarah asking if she wants to grab dinner tonight at 7pm, and set a reminder for 6:30pm to leave"
Bad: "Execute send_message_with_phone_number function"

Generate tasks now:
"""
```

### 3.2 Context Information Formatting

```python
def format_context_information(state_summary: Dict[str, Any]) -> str:
    """Format system state for prompt inclusion."""
    # pseudocode:
    # contacts_info = f"Contacts: {state_summary['contacts']['total_contacts']} contacts including {state_summary['contacts']['sample_names']}"
    # messages_info = f"Messages: {state_summary['messages']['total_messages']} messages, recent activity with {state_summary['messages']['recent_conversations']}"
    # settings_info = f"Device: WiFi {'on' if state_summary['settings']['wifi_enabled'] else 'off'}, Cellular {'on' if state_summary['settings']['cellular_enabled'] else 'off'}"
    # return combine all information into readable format
```

## 4. Usage Examples

### 4.1 Basic Usage

```python
# Initialize task generator
task_gen = OpenAITaskGeneratorRole(model_name="gpt-4")

# Generate 5 tasks using all available tools
tasks = task_gen.generate_tasks(num_tasks=5)

# Generate tasks for specific scenario
messaging_tasks = task_gen.generate_tasks(
    num_tasks=3,
    tool_filter=["send_message_with_phone_number", "search_contacts"],
    category=TaskCategory.COMMUNICATION,
    complexity=TaskComplexity.SIMPLE
)
```

### 4.2 Integration with Existing Scenario System

```python
def create_scenarios_from_generated_tasks(tasks: List[GeneratedTask]) -> Dict[str, Scenario]:
    """Convert generated tasks into Tool Sandbox scenarios."""
    # pseudocode:
    # scenarios = {}
    # for task in tasks:
    #     scenario_name = generate_scenario_name(task.description)
    #     scenario = create_base_scenario_with_task_message(task.description)
    #     scenarios[scenario_name] = scenario
    # return scenarios
```

## 5. Testing Strategy

### 5.1 Unit Tests

**File:** `tests/roles/task_generator_test.py`

```python
class TestTaskGeneratorRole:
    def test_get_available_tools_description(self):
        # Test tool filtering and description generation
        pass

    def test_get_system_state_summary(self):
        # Test state extraction from different database states
        pass

    def test_parse_and_validate_tasks(self):
        # Test parsing of various LLM response formats
        pass

    def test_task_validation(self):
        # Test validation of task structure and tool references
        pass

class TestOpenAITaskGenerator:
    def test_llm_call_with_retry(self):
        # Test API call and retry logic
        pass

class TestAnthropicTaskGenerator:
    def test_llm_call_with_retry(self):
        # Test API call and retry logic
        pass
```

### 5.2 Integration Tests

```python
class TestTaskGeneratorIntegration:
    def test_end_to_end_task_generation(self):
        # Test complete flow from context to generated tasks
        pass

    def test_with_different_system_states(self):
        # Test task generation with various database states
        pass

    def test_tool_filtering(self):
        # Test task generation with tool subsets
        pass
```

## 6. Error Handling and Edge Cases

### 6.1 LLM Response Handling

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

### 6.2 Validation and Fallbacks

```python
def validate_and_fix_tasks(tasks: List[Dict[str, Any]], available_tools: List[str]) -> List[GeneratedTask]:
    """Validate tasks and attempt to fix common issues."""
    # pseudocode:
    # for each task:
    # - check if required_tools exist in available_tools
    # - estimate steps if missing
    # - assign category if missing
    # - fix common formatting issues
    # - discard tasks that can't be fixed
```

## 7. Future Enhancements

### 7.1 Advanced Features (Post-MVP)

1. **Difficulty Progression**: Generate task sequences of increasing complexity
2. **User Persona Integration**: Generate tasks based on user profiles (student, professional, etc.)
3. **Temporal Awareness**: Generate time-sensitive tasks based on current time/date
4. **Multi-turn Task Chains**: Generate related task sequences that build on each other
5. **Constraint Satisfaction**: Ensure generated tasks can actually be completed with available tools

### 7.2 Performance Optimizations

1. **Template Caching**: Cache formatted prompt templates
2. **Batch Generation**: Generate multiple task sets in parallel
3. **Smart Retries**: Implement exponential backoff with jitter
4. **Response Caching**: Cache LLM responses for similar contexts

## 8. Dependencies and Requirements

### 8.1 New Dependencies
- No new external dependencies required (reuse existing OpenAI, Anthropic clients)

### 8.2 Internal Dependencies
- `tool_sandbox.common.tool_discovery.get_all_tools`
- `tool_sandbox.common.tool_conversion.get_tool_docs_natural_language`
- `tool_sandbox.common.execution_context.get_current_context`
- `tool_sandbox.roles.base_role.BaseRole`
- Existing retry mechanisms from agent implementations

## 9. File Structure Summary

```
tool_sandbox/
├── common/
│   └── execution_context.py          # Add TASK_GENERATOR role type
├── roles/
│   └── task_generator_role.py        # Main implementation (NEW)
├── cli/
│   └── utils.py                      # Add factory mappings (OPTIONAL)
└── tests/
    └── roles/
        └── task_generator_test.py    # Unit tests (NEW)
```

This implementation plan provides a comprehensive foundation for the task generator role while maintaining consistency with the existing Tool Sandbox architecture and patterns.
