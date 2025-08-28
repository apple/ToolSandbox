"""Tests for trajectory_parser.py functions."""

import json
import tempfile
from pathlib import Path

import pytest

from tool_sandbox.analysis.trajectory_parser import (
    _extract_tool_calls_from_sandbox,
    _preprocess_database_changes,
    parse_trajectory_for_goal_inference,
)
from tool_sandbox.analysis.trajectory_types import (
    DatabaseStateSnapshot,
    GoalInferenceToolCall,
    ParsedGoalInferenceTrajectory,
    ToolCallStep,
)
from tool_sandbox.common.execution_context import DatabaseNamespace


class TestExtractToolCallsFromSandbox:
    """Tests for _extract_tool_calls_from_sandbox function."""

    def test_extract_single_tool_call(self) -> None:
        """Test extracting a single tool call from SANDBOX database."""
        execution_context_data = {
            "_dbs": {
                "SANDBOX": [
                    {
                        "sandbox_message_index": 10,
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "call_test123_parameters = {'name': 'Alex'}\ncall_test123_response = search_contacts(**call_test123_parameters)",
                        "openai_tool_call_id": "call_test123",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    },
                    {
                        "sandbox_message_index": 11,
                        "sender": "EXECUTION_ENVIRONMENT",
                        "recipient": "AGENT",
                        "content": "[{'person_id': '123', 'name': 'Alex'}]",
                        "openai_tool_call_id": "call_test123",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    }
                ]
            }
        }

        result = _extract_tool_calls_from_sandbox(execution_context_data, 0, 20)

        assert len(result) == 1
        tool_call = result[0]
        assert isinstance(tool_call, GoalInferenceToolCall)
        assert tool_call.tool_name == "search_contacts"
        assert tool_call.arguments == {"name": "Alex"}
        assert tool_call.call_id == "call_test123"
        assert tool_call.sequence_index == 0
        assert tool_call.python_function_string == "search_contacts(name='Alex')"
        assert tool_call.result == "[{'person_id': '123', 'name': 'Alex'}]"

    def test_extract_multiple_tool_calls(self) -> None:
        """Test extracting multiple tool calls in sequence."""
        execution_context_data = {
            "_dbs": {
                "SANDBOX": [
                    {
                        "sandbox_message_index": 10,
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "call_1_parameters = {'name': 'Alex'}\ncall_1_response = search_contacts(**call_1_parameters)",
                        "openai_tool_call_id": "call_1",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    },
                    {
                        "sandbox_message_index": 11,
                        "sender": "EXECUTION_ENVIRONMENT",
                        "recipient": "AGENT",
                        "content": "result1",
                        "openai_tool_call_id": "call_1",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    },
                    {
                        "sandbox_message_index": 12,
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "call_2_parameters = {'phone_number': '+1234', 'content': 'Hello'}\ncall_2_response = send_message_with_phone_number(**call_2_parameters)",
                        "openai_tool_call_id": "call_2",
                        "openai_function_name": "send_message_with_phone_number",
                        "conversation_active": True
                    },
                    {
                        "sandbox_message_index": 13,
                        "sender": "EXECUTION_ENVIRONMENT",
                        "recipient": "AGENT",
                        "content": "result2",
                        "openai_tool_call_id": "call_2",
                        "openai_function_name": "send_message_with_phone_number",
                        "conversation_active": True
                    }
                ]
            }
        }

        result = _extract_tool_calls_from_sandbox(execution_context_data, 0, 20)

        assert len(result) == 2
        assert result[0].sequence_index == 0
        assert result[1].sequence_index == 1
        assert result[0].tool_name == "search_contacts"
        assert result[1].tool_name == "send_message_with_phone_number"

    def test_ignore_invalid_tool_calls(self) -> None:
        """Test that invalid tool calls are skipped."""
        execution_context_data = {
            "_dbs": {
                "SANDBOX": [
                    {
                        "sandbox_message_index": 10,
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "call_invalid_parameters = invalid_syntax\ncall_invalid_response = search_contacts(**call_invalid_parameters)",
                        "openai_tool_call_id": "call_invalid",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    },
                    {
                        "sandbox_message_index": 11,
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "call_valid_parameters = {'name': 'Alex'}\ncall_valid_response = search_contacts(**call_valid_parameters)",
                        "openai_tool_call_id": "call_valid",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    }
                ]
            }
        }

        result = _extract_tool_calls_from_sandbox(execution_context_data, 0, 20)

        assert len(result) == 1  # Only valid tool call
        assert result[0].call_id == "call_valid"

    def test_no_tool_calls(self) -> None:
        """Test SANDBOX database with no tool calls."""
        execution_context_data = {
            "_dbs": {
                "SANDBOX": [
                    {
                        "sandbox_message_index": 10,
                        "sender": "USER",
                        "recipient": "AGENT",
                        "content": "Hello",
                        "openai_tool_call_id": None,
                        "openai_function_name": None,
                        "conversation_active": True
                    },
                    {
                        "sandbox_message_index": 11,
                        "sender": "AGENT",
                        "recipient": "USER",
                        "content": "Hi there!",
                        "openai_tool_call_id": None,
                        "openai_function_name": None,
                        "conversation_active": True
                    }
                ]
            }
        }

        result = _extract_tool_calls_from_sandbox(execution_context_data, 0, 20)
        assert result == []


class TestPreprocessDatabaseChanges:
    """Tests for _preprocess_database_changes function."""

    def test_extract_single_namespace_diffs(self) -> None:
        """Test preprocessing database changes for single namespace."""
        execution_context_data = {
            "_dbs": {
                "CONTACT": [
                    {
                        "sandbox_message_index": 0,
                        "person_id": "123",
                        "name": "Alex",
                        "phone_number": "+1234"
                    },
                    {
                        "sandbox_message_index": 5,
                        "person_id": "456",
                        "name": "Bob",
                        "phone_number": "+5678"
                    }
                ]
            }
        }

        database_changes, initial_states = _preprocess_database_changes(execution_context_data, 1)

        assert "CONTACT" in database_changes
        assert "CONTACT" in initial_states

        # Initial state should contain the record at message index 0
        assert len(initial_states["CONTACT"]) == 1
        assert initial_states["CONTACT"][0]["person_id"] == "123"

        # Database changes should contain the new record at message index 5
        assert 5 in database_changes["CONTACT"]
        assert len(database_changes["CONTACT"][5]) == 1
        assert database_changes["CONTACT"][5][0]["person_id"] == "456"

    def test_filter_headguard_rows(self) -> None:
        """Test that headguard rows (all null except message index) are filtered out."""
        execution_context_data = {
            "_dbs": {
                "CONTACT": [
                    {
                        "sandbox_message_index": 0,
                        "person_id": None,
                        "name": None,
                        "phone_number": None
                    },
                    {
                        "sandbox_message_index": 0,
                        "person_id": "123",
                        "name": "Alex",
                        "phone_number": "+1234"
                    }
                ]
            }
        }

        database_changes, initial_states = _preprocess_database_changes(execution_context_data, 1)

        assert "CONTACT" in initial_states
        assert len(initial_states["CONTACT"]) == 1  # Only non-null record included
        assert initial_states["CONTACT"][0]["person_id"] == "123"

    def test_unknown_namespace_ignored(self) -> None:
        """Test that unknown database namespaces are ignored."""
        execution_context_data = {
            "_dbs": {
                "UNKNOWN_DB": [
                    {"sandbox_message_index": 0, "data": "value"}
                ],
                "CONTACT": [
                    {"sandbox_message_index": 0, "person_id": "123"}
                ]
            }
        }

        database_changes, initial_states = _preprocess_database_changes(execution_context_data, 1)

        assert "UNKNOWN_DB" not in database_changes
        assert "UNKNOWN_DB" not in initial_states
        assert "CONTACT" in initial_states


class TestParseTrajectoryForGoalInference:
    """Tests for parse_trajectory_for_goal_inference function."""

    def test_parse_complete_trajectory(self) -> None:
        """Test parsing a complete trajectory folder using execution_context only."""
        # Create execution context with SANDBOX database (new approach)
        execution_context_data = {
            "_dbs": {
                "SANDBOX": [
                    # Simulate first conversation ending
                    {
                        "sandbox_message_index": 5,
                        "sender": "USER",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "end_conversation()",
                        "openai_tool_call_id": None,
                        "openai_function_name": None,
                        "conversation_active": True
                    },
                    # Main conversation starts after index 5
                    {
                        "sandbox_message_index": 8,
                        "sender": "USER",
                        "recipient": "AGENT",
                        "content": "Find Alex's contact",
                        "openai_tool_call_id": None,
                        "openai_function_name": None,
                        "conversation_active": True
                    },
                    # Agent makes tool call
                    {
                        "sandbox_message_index": 10,
                        "sender": "AGENT",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "call_test123_parameters = {'name': 'Alex'}\ncall_test123_response = search_contacts(**call_test123_parameters)",
                        "openai_tool_call_id": "call_test123",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    },
                    # Execution environment responds
                    {
                        "sandbox_message_index": 11,
                        "sender": "EXECUTION_ENVIRONMENT",
                        "recipient": "AGENT",
                        "content": "[{'person_id': '123', 'name': 'Alex'}]",
                        "openai_tool_call_id": "call_test123",
                        "openai_function_name": "search_contacts",
                        "conversation_active": True
                    },
                    # End conversation
                    {
                        "sandbox_message_index": 15,
                        "sender": "USER",
                        "recipient": "EXECUTION_ENVIRONMENT",
                        "content": "end_conversation()",
                        "openai_tool_call_id": None,
                        "openai_function_name": None,
                        "conversation_active": True
                    }
                ],
                "CONTACT": [
                    {
                        "sandbox_message_index": 0,
                        "person_id": "123",
                        "name": "Alex"
                    },
                    {
                        "sandbox_message_index": 11,
                        "person_id": "456",
                        "name": "Bob"
                    }
                ]
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory_path = Path(temp_dir) / "test_scenario"
            trajectory_path.mkdir()

            # Only write execution_context.json (new approach doesn't need conversation.json)
            with (trajectory_path / "execution_context.json").open("w") as f:
                json.dump(execution_context_data, f)

            # Parse trajectory
            result = parse_trajectory_for_goal_inference(str(trajectory_path))

            assert isinstance(result, ParsedGoalInferenceTrajectory)
            assert result.scenario_name == "test_scenario"
            assert len(result.steps) == 1
            assert isinstance(result.steps[0], ToolCallStep)
            assert result.steps[0].tool_call.tool_name == "search_contacts"
            assert result.steps[0].tool_call.message_index == 10
            assert "CONTACT" in result.initial_database_state

    def test_missing_files_raises_error(self) -> None:
        """Test that missing execution_context.json raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory_path = Path(temp_dir) / "empty_scenario"
            trajectory_path.mkdir()

            with pytest.raises(FileNotFoundError, match="execution_context.json not found"):
                parse_trajectory_for_goal_inference(str(trajectory_path))
