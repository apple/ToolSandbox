# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for tool_sandbox.tools.messaging"""

import datetime
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import pytest
from polars.exceptions import DuplicateError

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    get_current_context,
    new_context,
)
from tool_sandbox.tools.contact import add_contact, search_contacts
from tool_sandbox.tools.messaging import search_messages, send_message_with_phone_number


@pytest.fixture
def self_contact() -> Dict[str, Any]:
    """Provide a test contact info for self

    Returns:

    """
    return {
        "person_id": str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name=__name__)),
        "name": "Tomas Haake",
        "phone_number": "+11233344455",
        "relationship": "self",
        "is_self": True,
    }


@pytest.fixture
def new_contact() -> Dict[str, str]:
    """Provide test contact info for new person

    Returns:
        name, phone number and relationship for test contact
    """
    return {
        "name": "Fredrik Thordendal",
        "phone_number": "+12453344098",
        "relationship": "friend",
    }


@pytest.fixture()
def new_messages() -> List[Dict[str, Optional[Union[float, str]]]]:
    """Provide test messages

    Returns:
        Rows of new messages
    """
    return [
        {
            "message_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="message_0")
            ),
            "sender_person_id": None,
            "sender_phone_number": "+18307976530",
            "recipient_person_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Tomas Haake")
            ),
            "recipient_phone_number": "+11233344455",
            "content": "Hey kid, you want some GPU?",
            "creation_timestamp": (
                datetime.datetime.now()
                - datetime.timedelta(days=3, hours=4, minutes=5, seconds=6)
            ).timestamp(),
        },
        {
            "message_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="message_1")
            ),
            "sender_person_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Tomas Haake")
            ),
            "sender_phone_number": "+11233344455",
            "recipient_person_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Homer S")
            ),
            "recipient_phone_number": "+10000000000",
            "content": "Things are proceeding as expected boss",
            "creation_timestamp": (
                datetime.datetime.now()
                - datetime.timedelta(hours=1, minutes=2, seconds=3)
            ).timestamp(),
        },
        {
            "message_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="message_2")
            ),
            "sender_person_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Homer S")
            ),
            "sender_phone_number": "+10000000000",
            "recipient_person_id": str(
                uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Tomas Haake")
            ),
            "recipient_phone_number": "+11233344455",
            "content": "Good, keep me posted",
            "creation_timestamp": (
                datetime.datetime.now() - datetime.timedelta(minutes=1, seconds=2)
            ).timestamp(),
        },
    ]


@pytest.fixture(scope="function", autouse=True)
def execution_context(
    self_contact: dict[str, Any],
    new_messages: list[dict[str, Optional[Union[float, str]]]],
) -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    # Set test context
    test_context = ExecutionContext()
    test_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT, rows=[self_contact]
    )
    test_context.add_to_database(
        namespace=DatabaseNamespace.MESSAGING,
        rows=new_messages,
    )
    with new_context(test_context):
        yield


def test_send_message_with_phone_number(
    new_contact: dict[str, str], self_contact: dict[str, Any]
) -> None:
    send_message_with_phone_number(
        phone_number=new_contact["phone_number"], content="Test"
    )
    current_context = get_current_context()
    messaging_database = current_context.get_database(
        namespace=DatabaseNamespace.MESSAGING
    )
    assert messaging_database["sender_person_id"][-1] == self_contact["person_id"]
    assert messaging_database["sender_phone_number"][-1] == self_contact["phone_number"]
    assert messaging_database["recipient_person_id"][-1] is None
    assert (
        messaging_database["recipient_phone_number"][-1] == new_contact["phone_number"]
    )
    assert messaging_database["content"][-1] == "Test"


def test_send_message_with_phone_number_multiple_self(
    new_contact: dict[str, str],
) -> None:
    # Multiple self
    test_context = get_current_context()
    test_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[
            {
                **new_contact,
                "is_self": True,
                "person_id": "test",
            }
        ],
    )
    with pytest.raises(DuplicateError):
        send_message_with_phone_number(
            phone_number=new_contact["phone_number"], content="Test"
        )


def test_send_message_with_name(
    new_contact: dict[str, str], self_contact: dict[str, Any]
) -> None:
    add_contact(**new_contact)
    # Suppose the user asks "Ask my man Fredrik how is the album coming along"
    # Start by finding the phone number for Fredrik
    matched_contacts = search_contacts(name="Fredrik")
    assert len(matched_contacts) == 1
    phone_number = matched_contacts[0]["phone_number"]
    # Send message
    send_message_with_phone_number(
        phone_number=phone_number, content="How is the album coming along"
    )
    # Check results
    current_context = get_current_context()
    messaging_database = current_context.get_database(
        namespace=DatabaseNamespace.MESSAGING
    )
    assert messaging_database["sender_person_id"][-1] == self_contact["person_id"]
    assert messaging_database["sender_phone_number"][-1] == self_contact["phone_number"]
    assert (
        messaging_database["recipient_person_id"][-1]
        == matched_contacts[0]["person_id"]
    )
    assert (
        messaging_database["recipient_phone_number"][-1] == new_contact["phone_number"]
    )
    assert messaging_database["content"][-1] == "How is the album coming along"


def test_search_messages(
    new_messages: list[dict[str, Optional[Union[float, str]]]],
) -> None:
    assert new_messages[-1:] == search_messages(
        message_id=str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="message_2"))
    )
    assert new_messages[-1:] == search_messages(
        sender_person_id=str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Homer S"))
    )
    assert new_messages[-1:] == search_messages(sender_phone_number="+10000000000")
    assert new_messages[:1] + new_messages[-1:] == search_messages(
        recipient_person_id=str(
            uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="Tomas Haake")
        )
    )
    assert new_messages[:1] + new_messages[-1:] == search_messages(
        recipient_phone_number="+11233344455"
    )
    assert new_messages[-1:] == search_messages(content="Good, keep me posted")
    assert new_messages[-2:] == search_messages(
        creation_timestamp_upperbound=datetime.datetime.now().timestamp(),
        creation_timestamp_lowerbound=(
            datetime.datetime.now() - datetime.timedelta(days=1)
        ).timestamp(),
    )
