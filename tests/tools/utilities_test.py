# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import datetime
from typing import Dict

import pytest
from pint.errors import DimensionalityError, UndefinedUnitError

from tool_sandbox.tools.utilities import (
    datetime_info_to_timestamp,
    search_holiday,
    timestamp_to_datetime_info,
    unit_conversion,
)


@pytest.fixture
def datetime_info() -> Dict[str, int]:
    """Provide test datetime info

    Returns:
        A dict of datatime info
    """
    return {"year": 2024, "month": 3, "day": 25, "hour": 22, "minute": 2, "second": 30}


@pytest.fixture
def timestamp(datetime_info: Dict[str, int]) -> float:
    """Provide test timestamp

    Returns:
        POSIX timestamp
    """
    return datetime.datetime(**datetime_info, tzinfo=None).timestamp()


def test_timestamp_to_datetime_info(
    timestamp: float, datetime_info: Dict[str, int]
) -> None:
    assert timestamp_to_datetime_info(timestamp) == {
        **datetime_info,
        "isoweekday": 1,
    }


def test_datetime_info_to_timestamp(
    timestamp: float, datetime_info: Dict[str, int]
) -> None:
    assert datetime_info_to_timestamp(**datetime_info) == timestamp


def test_unit_conversion() -> None:
    with pytest.raises(UndefinedUnitError):
        unit_conversion(amount=1.0, from_unit="BEEG", to_unit="Yoshi")

    with pytest.raises(DimensionalityError):
        unit_conversion(amount=1.0, from_unit="meter", to_unit="kg")

    assert (
        unit_conversion(amount=1.0, from_unit="meter", to_unit="mile")
        == 0.0006213711922373339
    )
    assert (
        pytest.approx(unit_conversion(amount=0.0, from_unit="degC", to_unit="degF"))
        == 32.0
    )


def test_search_holiday() -> None:
    # Non-existing holiday.
    assert search_holiday(holiday_name="Axlearn Day") is None
    # Holiday timestamp is dependent on system time zone.
    # Exact string match
    assert (
        search_holiday(holiday_name="Christmas Day", year=2024)
        == datetime.datetime(year=2024, month=12, day=25).timestamp()
    )
    # Partial match
    assert (
        search_holiday(holiday_name="new years", year=2024)
        == datetime.datetime(year=2024, month=1, day=1).timestamp()
    )
    # Test threshold accuracy
    assert search_holiday(holiday_name="new years eve", year=2024) is None
