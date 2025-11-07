# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for tool_sandbox.tools.contact"""

import uuid
from typing import Any, Dict, Iterator

import polars as pl
import pytest
from polars.exceptions import DuplicateError, NoDataError

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    get_current_context,
    new_context,
)
from tool_sandbox.tools.contact import (
    add_contact,
    modify_contact,
    remove_contact,
    search_contacts,
)


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


@pytest.fixture(scope="function", autouse=True)
def execution_context(self_contact: dict[str, Any]) -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    test_context = ExecutionContext()
    test_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT, rows=[self_contact]
    )
    with new_context(test_context):
        yield


def test_add_contact(new_contact: dict[str, str]) -> None:
    add_contact(**new_contact)
    current_context = get_current_context()
    contact_database = current_context.get_database(namespace=DatabaseNamespace.CONTACT)
    for k, v in new_contact.items():
        assert contact_database[k][1] == v
    with pytest.raises(DuplicateError):
        add_contact(**new_contact, is_self=True)


def test_modify_contact(new_contact: dict[str, str]) -> None:
    current_context = get_current_context()
    person_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="person_0"))
    current_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[{**new_contact, "person_id": person_id, "is_self": False}],
    )
    modify_contact(
        person_id=person_id,
        name="Fred",
        phone_number="+11234567890",
        relationship="coworker",
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.CONTACT).filter(
            pl.col("person_id") == person_id
        )["name"][-1]
        == "Fred"
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.CONTACT).filter(
            pl.col("person_id") == person_id
        )["phone_number"][-1]
        == "+11234567890"
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.CONTACT).filter(
            pl.col("person_id") == person_id
        )["relationship"][-1]
        == "coworker"
    )
    with pytest.raises(ValueError):
        modify_contact(person_id=person_id)
    with pytest.raises(NoDataError):
        modify_contact(person_id="fake id", name="Freddy")
    with pytest.raises(DuplicateError):
        modify_contact(person_id=person_id, is_self=True)
    current_context.add_to_database(
        namespace=DatabaseNamespace.CONTACT,
        rows=[{**new_contact, "person_id": person_id, "is_self": False}],
    )
    with pytest.raises(DuplicateError):
        modify_contact(person_id=person_id, name="Freddy")


def test_remove_contact(self_contact: dict[str, Any]) -> None:
    # Failure
    with pytest.raises(NoDataError):
        remove_contact(
            person_id=str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="wrong id"))
        )
    # Success
    remove_contact(person_id=self_contact["person_id"])
    current_context = get_current_context()
    contact_database = current_context.get_database(namespace=DatabaseNamespace.CONTACT)
    assert contact_database.is_empty()


def test_search_contacts(self_contact: dict[str, Any]) -> None:
    # Exact matching uuid
    assert [self_contact] == search_contacts(person_id=self_contact["person_id"])
    # Fuzzy match name
    assert [self_contact] == search_contacts(name="Tomas Nils Haake")
    # Exact match phone number
    assert [self_contact] == search_contacts(phone_number=self_contact["phone_number"])
    # Fuzzy match relationship
    assert [self_contact] == search_contacts(relationship="myself")
    # Exact match is_self
    assert [self_contact] == search_contacts(is_self=True)
    # No arguments
    with pytest.raises(ValueError):
        search_contacts()
    # Exact matching failure
    assert (
        len(
            search_contacts(
                person_id=str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="wrong id"))
            )
        )
        == 0
    )
    # Fuzzy matching failure
    assert len(search_contacts(relationship="friend")) == 0
