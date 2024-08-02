# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for tool_sandbox.common.utils"""

import datetime

import polars as pl
import pytest

from tool_sandbox.common.utils import (
    exact_match_filter_dataframe,
    fuzzy_match_filter_dataframe,
    is_close,
    range_filter_dataframe,
)


@pytest.fixture
def dataframe() -> pl.DataFrame:
    """
    Creates dataframe for matching filter testing

    Returns:
        Testing dataframe
    """
    return pl.DataFrame(
        {
            "name": ["John Wick", "Wick John", "Mike Portnoy"],
            "age": [10, 20, 30],
            "dob": [
                datetime.datetime(year=2000, month=10, day=2),
                datetime.datetime(year=2001, month=11, day=3),
                datetime.datetime(year=2002, month=12, day=4),
            ],
        }
    )


def test_exact_match_filter_dataframe(dataframe: pl.DataFrame) -> None:
    # Match success
    assert exact_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="John Wick"
    ).equals(dataframe[:1])
    # Match failure
    assert exact_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="john wick"
    ).is_empty()


def test_fuzzy_match_filter_dataframe(dataframe: pl.DataFrame) -> None:
    # Match success
    assert fuzzy_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="John Wick", threshold=100
    ).equals(dataframe[:1])
    # Casing invariant
    assert fuzzy_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="john wick", threshold=100
    ).equals(dataframe[:1])
    # Lower threshold
    assert fuzzy_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="John Wick", threshold=50
    ).equals(dataframe[:2])
    # Match failure
    assert fuzzy_match_filter_dataframe(
        dataframe=dataframe, column_name="name", value="John Petrucci", threshold=100
    ).is_empty()


def test_range_filter_dataframe(dataframe: pl.DataFrame) -> None:
    # Earlier
    assert range_filter_dataframe(
        dataframe=dataframe,
        column_name="dob",
        value=datetime.datetime(year=2002, month=12, day=5),
        value_delta=datetime.timedelta(days=-2),
    ).equals(dataframe[-1:])
    # Later
    assert range_filter_dataframe(
        dataframe=dataframe,
        column_name="dob",
        value=datetime.datetime(year=2000, month=9, day=30),
        value_delta=datetime.timedelta(days=5),
    ).equals(dataframe[:1])


def test_is_close() -> None:
    assert is_close(value=1.0, reference=0.8, atol=0.3)
    assert is_close(value=1.0, reference=1.0)
    assert not is_close(value=1.0, reference=0.8)
    assert is_close(value=1, reference=1)
