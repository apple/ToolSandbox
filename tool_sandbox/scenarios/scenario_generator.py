import datetime
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pathlib import Path

import polars as pl
import yaml
from attrs import define

from tool_sandbox.common.execution_context import DatabaseNamespace, ExecutionContext
from tool_sandbox.common.tool_conversion import get_tool_docs_natural_language
from tool_sandbox.common.tool_discovery import ToolBackend, ToolCategoryInfo, get_tool_categories_info
from tool_sandbox.common.utils import deterministic_uuid


class FormatError(Exception):
    """Exception raised when the task format is incorrect."""

    pass


# ! not exactly sure what will make tasks complex for goal inference.
class TaskComplexity(StrEnum):
    """Complexity of the task."""

    EASY = auto()
    MEDIUM = auto()
    HARD = auto()


@define
class GeneratedTask:
    """Container for a generated task."""

    task_id: str
    description: str
    required_tools: list[str]
    tools_category: list[str]

    def __str__(self) -> str:
        """Return a nicely formatted string representation."""
        return (
            f"Generated Task:\n"
            f"  ID: {self.task_id}\n"
            f"  Description: {self.description}\n"
            f"  Required Tools: {', '.join(self.required_tools)}\n"
            f"  Categories: {', '.join(self.tools_category)}"
        )


class TaskGenerator(ABC):
    """Base class for task generator roles."""

    def __init__(
        self,
        execution_context: ExecutionContext,
        max_retries: int = 3,
        populate_sample_data: bool = False,
    ) -> None:
        """Initialize the task generator role.

        Args:
            execution_context: Current system state
            max_retries: Maximum number of retries for task generation
            populate_sample_data: Whether to populate sample data for the execution context
        """
        self.all_tool_categories_info = get_tool_categories_info()
        self.execution_context = execution_context
        self.populate_sample_data = populate_sample_data
        self.max_retries = max_retries
        self.messages: list[dict[str, str]] = []

    def reset_messages(self) -> None:
        """Reset the messages."""
        self.messages = []

    def generate_task(
        self,
        allowed_tool_categories: list[str],
        preferred_tool_backend: ToolBackend = ToolBackend.DEFAULT,
    ) -> GeneratedTask:
        """Generate a single task based on tool categories and system state.

        Args:
            allowed_tool_categories: List of tool category names to focus on
            preferred_tool_backend: Which backend should be chosen for tools

        Returns:
            GeneratedTask: A single generated task
        """
        # 1. validate_tool_categories(tool_categories)
        self.reset_messages()
        allowed_tool_categories_info = self.validate_tool_categories(allowed_tool_categories)

        tools_description = self.get_tools_description(allowed_tool_categories_info)

        state_summary = self.get_state_summary(allowed_tool_categories_info)

        system_prompt, user_prompt = self.format_task_generation_prompt(
            tools_description, state_summary, allowed_tool_categories
        )

        # print(f"System prompt: {system_prompt}\n\n")
        # print(f"Task prompt: {user_prompt}\n\n")

        attempt = 0
        while attempt < self.max_retries:
            try:
                llm_response = self.model_inference(system_prompt, user_prompt)
                print(f"LLM response {attempt}: {llm_response}\n\n")
                parsed_task = self.parse_and_validate_task(llm_response, allowed_tool_categories_info)
            except FormatError as e:
                attempt += 1
                feedback_user_prompt = self.create_format_correction_prompt(response=llm_response, error_details=str(e))
                print(f"Feedback user prompt {attempt}: {feedback_user_prompt}\n\n")
                user_prompt = feedback_user_prompt
            except Exception:
                raise
            else:
                return parsed_task

        raise RuntimeError("Failed to generate task after max retries.")

    def create_format_correction_prompt(self, response: str, error_details: str) -> str:
        """Create a feedback user prompt for format correction.

        Args:
            response: The response from the model
            error_details: The details of the error

        Returns:
            str: The feedback user prompt
        """
        prompts_file = Path(__file__).parent / "format_correct_prompt.yaml"
        prompt_templates = yaml.safe_load(prompts_file.read_text())
        feedback_prompt_template: str = prompt_templates["format_correction"]
        return feedback_prompt_template.format(incorrect_response=response, error_explanation=error_details)

    def validate_tool_categories(self, tool_categories: list[str]) -> dict[str, ToolCategoryInfo]:
        """Validate the input tool categories and check if the category is valid.

        Args:
            tool_categories: List of tool category names to validate

        Returns:
            dict[str, ToolCategoryInfo]: Dictionary of valid tool categories and their information.

        Raises:
            ValueError: If any of the tool categories are invalid.
        """
        allowed_tool_categories_info = {}
        for category in tool_categories:
            if category not in self.all_tool_categories_info:
                raise ValueError(
                    f"Invalid tool category: {category}. Available categories: {self.all_tool_categories_info.keys()}"
                )
            else:
                allowed_tool_categories_info[category] = self.all_tool_categories_info[category]
        return allowed_tool_categories_info

    def get_tools_description(self, allowed_tool_categories_info: dict[str, ToolCategoryInfo]) -> str:
        """Get the description of tools in the required tool categories.

        Args:
            allowed_tool_categories_info: Dictionary of valid tool categories and their information.

        Returns:
            str: Prepared tool description for all available tools in the tool categories.
        """
        available_tools = {}
        for _, category_info in allowed_tool_categories_info.items():
            available_tools.update(category_info.tools)
        return get_tool_docs_natural_language(available_tools)

    def context_needs_population(self, databases: list[DatabaseNamespace]) -> list[DatabaseNamespace]:
        """Check if the execution context needs population.

        Args:
            databases: List of databases to check if they need population

        Returns:
            list[DatabaseNamespace]: List of databases that need population
        """
        databases_to_populate = []
        for database in databases:
            db = self.execution_context.get_database(database)
            if len(db) <= 1:
                databases_to_populate.append(database)
        return databases_to_populate

    def get_state_summary(self, allowed_tool_categories_info: dict[str, ToolCategoryInfo]) -> str:
        """Get the summary of the execution context state, populate sample data if needed.

        Args:
            allowed_tool_categories_info: Dictionary of valid tool categories and their information.

        Returns:
            str: The summary of the execution context
        """
        relevant_databases = [
            category.database for category in allowed_tool_categories_info.values() if category.database
        ]
        if self.populate_sample_data:
            databases_to_populate = self.context_needs_population(relevant_databases)
            self.populate_database(databases_to_populate)

        print(f"Relevant databases: {relevant_databases}\n\n")
        summary_parts = []
        for db_namespace in relevant_databases:
            summary = self._summarize_database(db_namespace)
            if summary:
                summary_parts.append(summary)
        return "\n".join(summary_parts) if summary_parts else "System State is empty"

    def format_task_generation_prompt(
        self, tools_description: str, state_summary: str, tool_categories: list[str]
    ) -> tuple[str, str]:
        """Format the task generation prompt.

        Args:
            tools_description: Description of the tools
            state_summary: Summary of the execution context state
            tool_categories: List of tool categories to use for the task

        Returns:
            tuple[str, str]: Tuple of system prompt and user prompt
        """
        prompts_file = Path(__file__).parent / "task_generation_prompts.yaml"
        prompt_templates = yaml.safe_load(prompts_file.read_text())

        system_prompt = prompt_templates["system_prompt"]
        task_prompt_template = prompt_templates["task_prompt"]
        category_descriptions = prompt_templates["category_descriptions"]
        tool_categories_descriptions = [
            f"{category}: {category_descriptions[category]}" for category in tool_categories
        ]

        user_prompt = task_prompt_template.format(
            tools_description=tools_description,
            state_summary=state_summary,
            tool_categories=", ".join(tool_categories_descriptions),
        )

        return system_prompt, user_prompt

    @abstractmethod
    def model_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call the model to generate a task.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt
        """
        raise NotImplementedError("model_inference method must be implemented by subclasses.")

    # FIXME: fix the complexity of this method.
    def parse_and_validate_task(  # noqa: C901
        self, llm_response: str, allowed_tool_categories_info: dict[str, ToolCategoryInfo]
    ) -> GeneratedTask:
        """Parse and validate the task.

        Args:
            llm_response: The task string generated by the model
            allowed_tool_categories_info: The allowed tool categories info

        Returns:
            GeneratedTask: A single generated task

        Raises:
            FormatError: If the task format is incorrect
        """
        lines = llm_response.strip().split("\n")
        task_description = None
        trajectory = None
        categories = None

        for line in lines:
            line = line.strip()
            if line.startswith("TASK:"):
                task_description = line[len("TASK:") :].strip()
            elif line.startswith("TRAJECTORY:"):
                trajectory = line[len("TRAJECTORY:") :].strip()
            elif line.startswith("CATEGORIES:"):
                categories = line[len("CATEGORIES:") :].strip()

        # validate required fields
        if not task_description:
            raise FormatError("Missing TASK: TASK field is required.")
        if not trajectory:
            raise FormatError("Missing TRAJECTORY: TRAJECTORY field is required.")
        if not categories:
            raise FormatError("Missing CATEGORIES: CATEGORIES field is required.")

        required_tools = [tool.strip() for tool in trajectory.split(",") if tool.strip()]
        tools_category = [category.strip() for category in categories.split(",") if category.strip()]

        allowed_tools = {}
        for category_info in allowed_tool_categories_info.values():
            allowed_tools.update(category_info.tools)

        invalid_tools = [tool for tool in required_tools if tool not in allowed_tools]
        if invalid_tools:
            raise FormatError(f"Invalid tools: {invalid_tools}. Available tools: {allowed_tools.keys()}")

        invalid_categories = [category for category in tools_category if category not in allowed_tool_categories_info]
        if invalid_categories:
            raise FormatError(
                f"Invalid categories: {invalid_categories}. Available categories: {allowed_tool_categories_info.keys()}"
            )

        task_payload = f"task_{task_description[:50]}_{','.join(required_tools)}"
        task_id = deterministic_uuid(payload=task_payload)

        return GeneratedTask(
            task_id=task_id,
            description=task_description,
            required_tools=required_tools,
            tools_category=tools_category,
        )

    # ! fill in the implementation of this method.
    def populate_database(self, databases: list[DatabaseNamespace]) -> None:  # noqa: B027
        """Populate sample data for the execution context.

        Args:
            databases: List of databases to populate sample data for
        """
        pass

    def _summarize_database(self, database: DatabaseNamespace) -> str:
        """Summarize the database.

        Args:
            database: Database to summarize

        Returns:
            str: Summary of the database state
        """
        db = self.execution_context.get_database(database)
        if len(db) <= 1:  # Only headguard
            return ""

        match database:
            case DatabaseNamespace.CONTACT:
                return self._summarize_contact_database(db)
            case DatabaseNamespace.MESSAGING:
                return self._summarize_messaging_database(db)
            case DatabaseNamespace.REMINDER:
                return self._summarize_reminder_database(db)
            case DatabaseNamespace.SETTING:
                return self._summarize_setting_database(db)
            case DatabaseNamespace.SANDBOX:
                return self._summarize_sandbox_database(db)
            case _:
                # Fallback for unknown database types
                return f"{database.value} database has {len(db) - 1} entries."

    def _summarize_contact_database(self, db: pl.DataFrame) -> str:
        """Summarize contact database content.

        Args:
            db: Contact database

        Returns:
            str: Summary of the contact database state
        """
        contacts = []
        for i in range(1, len(db)):  # Skip headguard row
            name = db["name"][i]
            relationship = db["relationship"][i]
            is_self = db["is_self"][i]
            if not is_self:  # Only summarize other contacts
                contacts.append(f"{name} ({relationship})")

        if contacts:
            summary = f"You have {len(contacts)} contacts including {', '.join(contacts[:3])}"
            if len(contacts) > 3:
                summary += f" and {len(contacts) - 3} others."
            return summary
        return "You have no other contacts."

    def _summarize_messaging_database(self, db: pl.DataFrame) -> str:
        """Summarize messaging database content.

        Args:
            db: Messaging database

        Returns:
            str: Summary of the messaging database state
        """
        recent_messages = []
        current_time = datetime.datetime.now().timestamp()

        # Get up to 3 most recent messages, skipping headguard
        for i in range(max(1, len(db) - 3), len(db)):
            content = db["content"][i]
            timestamp = db["creation_timestamp"][i]
            sender_phone = db["sender_phone_number"][i]

            hours_ago = (current_time - timestamp) / 3600

            sender_name = "Unknown"
            # Simple lookup for sender name from contacts
            contact_db = self.execution_context.get_database(DatabaseNamespace.CONTACT)
            for j in range(1, len(contact_db)):
                if contact_db["phone_number"][j] == sender_phone:
                    sender_name = contact_db["name"][j]
                    break

            if hours_ago < 1:
                time_desc = "just now"
            elif hours_ago < 24:
                time_desc = f"{int(hours_ago)} hours ago"
            else:
                time_desc = f"{int(hours_ago // 24)} days ago"

            recent_messages.append(f"{sender_name}: '{content}' ({time_desc})")

        if recent_messages:
            return f"Recent messages: {'; '.join(recent_messages)}."
        return "No recent messages."

    def _summarize_reminder_database(self, db: pl.DataFrame) -> str:
        """Summarize reminder database content.

        Args:
            db: Reminder database

        Returns:
            str: Summary of the reminder database state
        """
        reminders = []
        current_time = datetime.datetime.now().timestamp()

        for i in range(1, len(db)):  # Skip headguard
            content = db["content"][i]
            reminder_time = db["reminder_timestamp"][i]

            hours_diff = (reminder_time - current_time) / 3600

            if hours_diff < -24:
                status = "overdue"
            elif hours_diff < 0:
                status = "recently due"
            elif hours_diff < 1:
                status = "due soon"
            else:
                status = "upcoming"

            reminders.append(f"{status} reminder: '{content}'")

        if reminders:
            summary = f"You have {len(reminders)} reminders: {'; '.join(reminders[:2])}"
            if len(reminders) > 2:
                summary += f" and {len(reminders) - 2} others."
            return summary
        return "You have no reminders."

    def _summarize_setting_database(self, db: pl.DataFrame) -> str:
        """Summarize device settings.

        Args:
            db: Setting database

        Returns:
            str: Summary of the setting database state
        """
        settings_info = []
        # Assuming the first actual row (index 1) contains the current settings
        if len(db) > 1:
            wifi = db["wifi"][1]
            cellular = db["cellular"][1]
            location = db["location_service"][1]
            low_battery = db["low_battery_mode"][1]

            if low_battery:
                settings_info.append("low battery mode is ON")
            if not wifi:
                settings_info.append("WiFi is OFF")
            if not cellular:
                settings_info.append("cellular is OFF")
            if not location:
                settings_info.append("location services are OFF")

            if settings_info:
                return f"Device settings: {', '.join(settings_info)}."
            return "Device settings: all services are enabled."
        return "No device settings configured."

    def _summarize_sandbox_database(self, db: pl.DataFrame) -> str:
        """Summarize sandbox/conversation database content.

        Args:
            db: Sandbox database

        Returns:
            str: Summary of the sandbox database state
        """
        if len(db) <= 1:
            return "No conversation history."

        # Count messages by role
        conversation_count = len(db) - 1  # Exclude headguard
        return f"Conversation history has {conversation_count} messages."
