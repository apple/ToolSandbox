# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
from typing import Any

import pytest

from tool_sandbox.tools.rapid_api_search_tools import (
    convert_currency,
    search_lat_lon,
    search_location_around_lat_lon,
    search_stock,
    search_weather_around_lat_lon,
)

if "RAPID_API_KEY" not in os.environ:
    pytest.skip(
        "Tests intended for local mac development with Rapid API access only.",
        allow_module_level=True,
    )


def test_convert_currency() -> None:
    with pytest.raises(ValueError):
        convert_currency(amount=1.5, from_currency_code="$", to_currency_code="@")
    # Separate check for integer amount, as a query with "amount=1.0" fails (!).
    convert_currency(amount=1, from_currency_code="chf", to_currency_code="eur")
    convert_currency(amount=1.5, from_currency_code="usd", to_currency_code="cny")


def test_search_lat_lon() -> None:
    # Coordinates out of valid range.
    with pytest.raises(ValueError):
        assert search_lat_lon(latitude=1000.0, longitude=1000.0) is None
    # North pole, might deliver a location result in future?
    assert search_lat_lon(latitude=90.0, longitude=180.0) is None
    # Valid address.
    assert (
        search_lat_lon(latitude=37.334606, longitude=-122.009102)
        == "One Apple Park Way, Cupertino, CA 95014, USA"
    )


def test_search_location_around_lat_lon() -> None:
    assert not search_location_around_lat_lon(location="?")
    apple_park_search_result: dict[str, Any] = search_location_around_lat_lon(
        location="Apple Park"
    )[0]
    for key in [
        "review_count",
        "types",
        "price_level",
        "working_hours",
        "description",
        "rating",
        "state",
    ]:
        apple_park_search_result.pop(key)
    assert apple_park_search_result == {
        "phone_number": "+14089961010",
        "name": "Apple Park",
        "full_address": "Apple Park, One Apple Park Way, Cupertino, CA 95014",
        "latitude": 37.334643799999995,
        "longitude": -122.008972,
        "timezone": "America/Los_Angeles",
        "website": "http://www.apple.com/",
        "city": "Cupertino, CA",
    }


def test_search_weather_around_lat_lon() -> None:
    with pytest.raises(ValueError):
        search_weather_around_lat_lon(days=-1)
    assert "current_temperature" in search_weather_around_lat_lon(
        latitude=37.334606, longitude=-122.009102
    )
    assert "current_temperature" not in search_weather_around_lat_lon(
        days=1, latitude=37.334606, longitude=-122.009102
    )


def test_search_stock() -> None:
    assert search_stock(query="Beeg Yoshi") is None
    assert search_stock(query="Apple")["symbol"] == "AAPL"
