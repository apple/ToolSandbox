# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Helper function that supports tool_trace_dependant_similarity. Extract certain values from a tool trace."""

from typing import Any, Dict, List, Literal

from typing_extensions import Protocol


class ToolTraceExtractorType(Protocol):
    """Callable type def for tool trace extraction functions.

    Takes a tool_trace as input, returns a List of possible normalizations of extracted dictionary values

    Errors should be caught inside the extractor, assuming input types could be invalid
    """

    def __call__(
        self,
        tool_trace: Dict[Literal["tool_name", "arguments", "result"], Any],
    ) -> List[Dict[str, Any]]:
        """Extract values from a tool trace.

        Args:
            tool_trace: The tool trace to extract values from.

        Returns:
            A list of possible normalizations of extracted dictionary values.
        """
        ...


def default_value_normalization(value: Any) -> List[str]:  # noqa: ANN401
    """Normalize objects into a List of surface forms. Normalization strategy varies depending on datatype.

    Args:
        value:  Value to normalize

    Returns:
        List of normalized values
    """
    if isinstance(value, float):
        return [
            f"{value:.0f}",
            f"{value:.1f}",
            f"{value:.2f}",
            f"{value:.3f}",
        ]

    return [str(value)]


def search_weather_around_lat_lon_temperature_extractor(
    tool_trace: Dict[Literal["tool_name", "arguments", "result"], Any],
) -> List[Dict[str, Any]]:
    """Extractor that extracts temperate from search_weather_around_lat_lon tool trace.

    Args:
        tool_trace:     tool_trace of search_weather_around_lat_lon

    Returns:
        a list of dict containing possible normalized forms of the temperature
    """
    try:
        temperature: float = float(tool_trace["result"]["current_temperature"])
        # Normalize into different precisions
        return [
            {"temperature": normalized_temperature}
            for normalized_temperature in default_value_normalization(temperature)
        ]
    except (KeyError, TypeError):
        return []


def result_to_timestamp0_extractor(
    tool_trace: dict[str, Any],
) -> list[dict[str, float]]:
    """Extract the timestamp_0 from a tool trace.

    Args:
        tool_trace: The tool trace to extract the timestamp_0 from.

    Returns:
        A list of dictionaries containing the timestamp_0.
    """
    return [{"timestamp_0": float(tool_trace["result"])}]


def result_to_timestamp1_extractor(
    tool_trace: dict[str, Any],
) -> list[dict[str, float]]:
    """Extract the timestamp_1 from a tool trace.

    Args:
        tool_trace: The tool trace to extract the timestamp_1 from.

    Returns:
        A list of dictionaries containing the timestamp_1.
    """
    return [{"timestamp_1": float(tool_trace["result"])}]


def days_extractor(tool_trace: dict[str, Any]) -> list[dict[str, int]]:
    """Extract the days from a tool trace.

    Args:
        tool_trace: The tool trace to extract the days from.

    Returns:
        A list of dictionaries containing the days.
    """
    return [{"days": int(tool_trace["result"]["days"])}]


def lat_lon_dict_extractor(tool_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the latitude and longitude from a tool trace.

    Args:
        tool_trace: The tool trace to extract the latitude and longitude from.

    Returns:
        A list of dictionaries containing the latitude and longitude.
    """
    return [{k: tool_trace["result"][k] for k in ["latitude", "longitude"]}]


def current_temperature_extractor(tool_trace: dict[str, Any]) -> list[dict[str, float]]:
    """Extract the current temperature from a tool trace.

    Args:
        tool_trace: The tool trace to extract the current temperature from.

    Returns:
        A list of dictionaries containing the current temperature.
    """
    return [
        {
            "amount": float(tool_trace["result"]["current_temperature"]),
        }
    ]


def result_to_temperature_extractor(
    tool_trace: dict[str, Any],
) -> list[dict[str, str]]:
    """Extract the temperature from a tool trace.

    Args:
        tool_trace: The tool trace to extract the temperature from.

    Returns:
        A list of dictionaries containing the temperature.
    """
    return [{"temperature": x} for x in default_value_normalization(float(tool_trace["result"]))]


def result_to_reminder_timestamp_lowerbound_extractor(
    tool_trace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract the reminder timestamp lowerbound from a tool trace.

    Args:
        tool_trace: The tool trace to extract the reminder timestamp lowerbound from.

    Returns:
        A list of dictionaries containing the reminder timestamp lowerbound.
    """
    return [{"reminder_timestamp_lowerbound": float(tool_trace["result"])}]


def min_temperature_amount_extractor(
    tool_trace: dict[str, Any],
) -> list[dict[str, float]]:
    """Extract the minimum temperature from a tool trace.

    Args:
        tool_trace: The tool trace to extract the minimum temperature from.

    Returns:
        A list of dictionaries containing the minimum temperature.
    """
    return [
        {
            "amount": float(tool_trace["result"]["min_temperature"]),
        }
    ]


def min_temperature_extractor(tool_trace: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the minimum temperature from a tool trace.

    Args:
        tool_trace: The tool trace to extract the minimum temperature from.

    Returns:
        A list of dictionaries containing the minimum temperature.
    """
    return [{"min_temperature": x} for x in default_value_normalization(float(tool_trace["result"]))]
