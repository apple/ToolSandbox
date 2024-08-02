# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import inspect
from textwrap import dedent

import pytest

from tool_sandbox.common.execution_context import (
    ScenarioCategories,
    new_context_with_attribute,
)
from tool_sandbox.common.tool_conversion import (
    convert_to_openai_tool,
    maybe_scramble_arg_description,
    maybe_scramble_arg_type,
    maybe_scramble_tool_description,
)
from tool_sandbox.common.tool_discovery import ToolBackend, get_all_tools
from tool_sandbox.tools.contact import add_contact, modify_contact


@pytest.mark.parametrize("tool_backend", ToolBackend)
def test_converting_all_tools(tool_backend: ToolBackend) -> None:
    """Ensure that all our tools can be converted to the OpenAI tool format."""
    name_to_tool = get_all_tools(preferred_tool_backend=tool_backend)
    for tool in name_to_tool.values():
        openai_tool = convert_to_openai_tool(tool)
        assert "function" == openai_tool["type"]
        function = openai_tool["function"]
        assert function is not None
        assert function["name"] is not None
        assert function["description"] is not None
        assert "object" == function["parameters"]["type"]

        # Ensure that each parameter has a type and a description.
        for param in function["parameters"]["properties"].values():
            assert param["type"] is not None
            assert param["description"] is not None

        # Ensure that all required parameters are part of the properties entry.
        for required_param_name in function["parameters"]["required"]:
            assert function["parameters"]["properties"][required_param_name] is not None


# Make sure some of our more complicated tools are being converted as expected
def test_convert_to_openai_tool() -> None:
    # Strip None
    assert convert_to_openai_tool(add_contact) == {
        "type": "function",
        "function": {
            "name": "add_contact",
            "description": "Add a new contact person to contact database. Entries with identical information "
            "are allowed,\nand will be assigned different person_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of contact person"},
                    "phone_number": {
                        "type": "string",
                        "description": "Phone number of contact person",
                    },
                    "relationship": {
                        "type": "string",
                        "description": "Optional, relationship between user and this contact",
                    },
                    "is_self": {
                        "type": "boolean",
                        "description": "Optional. Defaults to False. "
                        "Should be set to True if the contact corresponds to the current user.",
                    },
                },
                "required": ["name", "phone_number"],
            },
        },
    }
    # Strip NotGiven
    assert convert_to_openai_tool(modify_contact) == {
        "type": "function",
        "function": {
            "name": "modify_contact",
            "description": "Modify a contact entry with new information provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {
                        "type": "string",
                        "description": "String unique identifier for the contact person",
                    },
                    "name": {
                        "type": "string",
                        "description": "New name for the person",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "New phone number for the person",
                    },
                    "relationship": {
                        "type": "string",
                        "description": "New relationship for the person",
                    },
                    "is_self": {
                        "type": "boolean",
                        "description": "Optional. Defaults to False. "
                        "Should be set to True if the contact corresponds to the current user.",
                    },
                },
                "required": ["person_id"],
            },
        },
    }
    # Remove description
    with new_context_with_attribute(
        tool_augmentation_list=[ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED]
    ):
        assert convert_to_openai_tool(add_contact) == {
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of contact person",
                        },
                        "phone_number": {
                            "type": "string",
                            "description": "Phone number of contact person",
                        },
                        "relationship": {
                            "type": "string",
                            "description": "Optional, relationship between user and this contact",
                        },
                        "is_self": {
                            "type": "boolean",
                            "description": "Optional. Defaults to False. "
                            "Should be set to True if the contact corresponds to the current user.",
                        },
                    },
                    "required": ["name", "phone_number"],
                },
            },
        }
    # Remove argument description
    with new_context_with_attribute(
        tool_augmentation_list=[ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED]
    ):
        assert convert_to_openai_tool(add_contact) == {
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "Add a new contact person to contact database. "
                "Entries with identical information are allowed,"
                "\nand will be assigned different person_ids.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                        },
                        "phone_number": {
                            "type": "string",
                        },
                        "relationship": {
                            "type": "string",
                        },
                        "is_self": {
                            "type": "boolean",
                        },
                    },
                    "required": ["name", "phone_number"],
                },
            },
        }
    # Remove argument type
    with new_context_with_attribute(
        tool_augmentation_list=[ScenarioCategories.ARG_TYPE_SCRAMBLED]
    ):
        assert convert_to_openai_tool(add_contact) == {
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "Add a new contact person to contact database. Entries with identical information "
                "are allowed,\nand will be assigned different person_ids.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"description": "Name of contact person"},
                        "phone_number": {
                            "description": "Phone number of contact person",
                        },
                        "relationship": {
                            "description": "Optional, relationship between user and this contact",
                        },
                        "is_self": {
                            "description": "Optional. Defaults to False. "
                            "Should be set to True if the contact corresponds to the current user.",
                        },
                    },
                    "required": ["name", "phone_number"],
                },
            },
        }


def test_maybe_scramble_tool_description() -> None:
    docstring = """
    Args:
        name:           Name of contact person
        phone_number:   Phone number of contact person
        relationship:   Optional, relationship between user and this contact
        is_self:        Optional. Defaults to False. Should be set to True if the contact corresponds to the current user.

    Returns:
        String format unique identifier for the contact person, this can be passed to other functions
        which require a unique identifier for this contact person

    Raises:
        DuplicateError:     When self entry already exists, and we are adding a new one
    """
    # Without context
    extracted_docstring = inspect.getdoc(add_contact)
    assert extracted_docstring is not None
    assert (
        maybe_scramble_tool_description(
            extracted_docstring,
            tool_augmentation_list=[ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED],
        )
        == dedent(docstring).strip()
    )


def test_maybe_scramble_arg_description() -> None:
    docstring = """
    Add a new contact person to contact database. Entries with identical information are allowed,
    and will be assigned different person_ids.

    Args:

    Returns:
        String format unique identifier for the contact person, this can be passed to other functions
        which require a unique identifier for this contact person

    Raises:
        DuplicateError:     When self entry already exists, and we are adding a new one
    """
    # Without context
    extracted_docstring = inspect.getdoc(add_contact)
    assert extracted_docstring is not None
    assert (
        maybe_scramble_arg_description(
            extracted_docstring,
            tool_augmentation_list=[ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED],
        )
        == dedent(docstring).strip()
    )


def test_maybe_scramble_arg_type() -> None:
    for parameter in maybe_scramble_arg_type(
        list(inspect.signature(add_contact).parameters.values()),
        tool_augmentation_list=[ScenarioCategories.ARG_TYPE_SCRAMBLED],
    ):
        assert parameter.annotation is type(None)
