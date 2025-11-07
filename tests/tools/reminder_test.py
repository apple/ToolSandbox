# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import datetime
import uuid
from typing import Dict, Iterator, Union

import polars as pl
import pytest
from polars.exceptions import DuplicateError, NoDataError

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    get_current_context,
    new_context,
)
from tool_sandbox.tools.reminder import (
    add_reminder,
    modify_reminder,
    remove_reminder,
    search_reminder,
)


@pytest.fixture
def new_reminder() -> Dict[str, Union[str, float]]:
    """Provides a dictionary containing information about a new reminder entry

    Returns:

    """
    return {
        "content": "Return chocolate milk",
        "reminder_timestamp": (
            datetime.datetime.now()
            + datetime.timedelta(days=4, hours=4, minutes=5, seconds=6)
        ).timestamp(),
        "latitude": 37.3237926356735,
        "longitude": -122.03961770355414,
    }


@pytest.fixture(scope="function", autouse=True)
def execution_context() -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    # Set test context
    test_context = ExecutionContext()
    test_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=[
            {
                "reminder_id": str(
                    uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_0")
                ),
                "content": "Write tests",
                "creation_timestamp": (
                    datetime.datetime.now()
                    - datetime.timedelta(days=10, hours=1, minutes=2, seconds=3)
                ).timestamp(),
                "reminder_timestamp": (
                    datetime.datetime.now()
                    - datetime.timedelta(days=8, hours=4, minutes=5, seconds=6)
                ).timestamp(),
                "latitude": None,
                "longitude": None,
            },
            {
                "reminder_id": str(
                    uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1")
                ),
                "content": "Buy chocolate milk",
                "creation_timestamp": (
                    datetime.datetime.now()
                    - datetime.timedelta(hours=1, minutes=2, seconds=3)
                ).timestamp(),
                "reminder_timestamp": (
                    datetime.datetime.now()
                    + datetime.timedelta(days=2, hours=4, minutes=5, seconds=6)
                ).timestamp(),
                "latitude": 37.3237926356735,
                "longitude": -122.03961770355414,
            },
        ],
    )
    with new_context(test_context):
        yield


def test_add_reminder(new_reminder: Dict[str, Union[str, float]]) -> None:
    add_reminder(**new_reminder)
    current_context = get_current_context()
    reminder_database = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    )
    assert reminder_database["content"][-1] == new_reminder["content"]
    assert (
        reminder_database["reminder_timestamp"][-1]
        == new_reminder["reminder_timestamp"]
    )
    assert reminder_database["latitude"][-1] == new_reminder["latitude"]
    assert reminder_database["longitude"][-1] == new_reminder["longitude"]


def test_remove_reminder() -> None:
    current_context = get_current_context()
    reminder_database = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    )
    assert not reminder_database.filter(
        pl.col("reminder_id")
        == str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    ).is_empty()
    # Successful
    remove_reminder(
        reminder_id=str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    )
    reminder_database = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    )
    assert reminder_database.filter(
        pl.col("reminder_id")
        == str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    ).is_empty()
    # Error
    with pytest.raises(NoDataError):
        remove_reminder("This should raise error")


def test_modify_reminder() -> None:
    current_context = get_current_context()
    reminder_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    creation_timestamp = current_context.get_database(
        namespace=DatabaseNamespace.REMINDER
    ).filter(pl.col("reminder_id") == reminder_id)["creation_timestamp"][-1]
    modify_reminder(
        reminder_id=reminder_id,
        content="Test",
    )
    # Content should change
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["content"][-1]
        == "Test"
    )
    # Creation timestamp should change
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["creation_timestamp"][-1]
        > creation_timestamp
    )
    new_timestamp = datetime.datetime.now().timestamp()
    modify_reminder(
        reminder_id=reminder_id,
        reminder_timestamp=new_timestamp,
        latitude=None,
        longitude=None,
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["reminder_timestamp"][-1]
        == new_timestamp
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["latitude"][-1]
        is None
    )
    assert (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER).filter(
            pl.col("reminder_id") == reminder_id
        )["longitude"][-1]
        is None
    )
    # Error
    with pytest.raises(ValueError):
        modify_reminder(reminder_id=reminder_id)
    with pytest.raises(NoDataError):
        modify_reminder(
            reminder_id="This should raise error", content="This should raise error"
        )
    current_context.add_to_database(
        namespace=DatabaseNamespace.REMINDER,
        rows=current_context.get_database(
            namespace=DatabaseNamespace.REMINDER
        ).to_dicts()[-1:],
    )
    with pytest.raises(DuplicateError):
        modify_reminder(
            reminder_id=reminder_id,
            content="This should raise error",
        )


def test_search_reminder() -> None:
    current_context = get_current_context()
    reminder_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="reminder_1"))
    target_reminder = (
        current_context.get_database(namespace=DatabaseNamespace.REMINDER)
        .filter(pl.col("reminder_id") == reminder_id)
        .to_dicts()[0]
    )
    assert [target_reminder] == search_reminder(reminder_id=reminder_id)
    assert [target_reminder] == search_reminder(content=target_reminder["content"])
    assert [target_reminder] == search_reminder(
        creation_timestamp_lowerbound=target_reminder["creation_timestamp"] - 10,
        creation_timestamp_upperbound=target_reminder["creation_timestamp"] + 10,
    )
    assert [target_reminder] == search_reminder(
        reminder_timestamp_lowerbound=target_reminder["reminder_timestamp"] - 10,
        reminder_timestamp_upperbound=target_reminder["reminder_timestamp"] + 10,
    )
    assert [target_reminder] == search_reminder(
        latitude=target_reminder["latitude"],
    )
    assert [target_reminder] == search_reminder(
        longitude=target_reminder["longitude"],
    )
    # No arguments
    with pytest.raises(ValueError):
        search_reminder()
    # Exact matching failure
    assert (
        len(
            search_reminder(
                reminder_id=str(
                    uuid.uuid5(namespace=uuid.NAMESPACE_URL, name="wrong id")
                )
            )
        )
        == 0
    )
    # Fuzzy matching failure
    assert len(search_reminder(content="Hello")) == 0
    # Range matching failure
    assert (
        len(
            search_reminder(
                reminder_timestamp_lowerbound=target_reminder["reminder_timestamp"]
                + 10,
            )
        )
        == 0
    )
