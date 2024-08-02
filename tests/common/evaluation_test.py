# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import json
from typing import Any, Dict

import polars as pl
import pytest

from tool_sandbox.common.evaluation import (
    ColumnSimilarityMeasureType,
    addition_similarity,
    column_exact_match_similarity,
    column_rouge_l_similarity,
    column_tool_trace_exact_match_similarity,
    snapshot_similarity,
    tool_trace_dependant_similarity,
)
from tool_sandbox.common.execution_context import DatabaseNamespace, ExecutionContext


@pytest.fixture
def reference_snapshot() -> pl.DataFrame:
    """Testing snapshot

    Returns:
        A Dataframe object representing a snapshot
    """
    return pl.DataFrame(
        [
            {"sandbox_message_index": 0, "content": "Hello there", "num": 0},
        ]
    )


@pytest.fixture
def snapshot() -> pl.DataFrame:
    """Testing snapshot

    Returns:
        A Dataframe object representing a snapshot
    """
    return pl.DataFrame(
        [
            {"sandbox_message_index": 1, "content": "Hello there", "num": 0},
            {"sandbox_message_index": 1, "content": "Hey there", "num": 1},
        ]
    )


@pytest.fixture
def column_similarities() -> Dict[str, ColumnSimilarityMeasureType]:
    """Testing column similarities

    Returns:

    """
    return {
        "content": column_rouge_l_similarity,
        "num": column_exact_match_similarity,
    }


def test_snapshot_similarity(
    reference_snapshot: pl.DataFrame,
    snapshot: pl.DataFrame,
    column_similarities: Dict[str, ColumnSimilarityMeasureType],
) -> None:
    # Different row count
    assert (
        snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hello there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    # Different columns
    assert (
        snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {
                        "sandbox_message_index": 0,
                        "content": "Hello there",
                        "wrong_column": "?",
                    },
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    # Subset columns
    assert (
        snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"num": 0},
                    {"num": 1},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 1
    )
    # Similarity shouldn't be affected by row orders
    assert (
        0
        < snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 0},
                    {"content": "Hi there", "num": 1},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == snapshot_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        < 1
    )


def test_addition_similarity(
    reference_snapshot: pl.DataFrame,
    snapshot: pl.DataFrame,
    column_similarities: Dict[str, ColumnSimilarityMeasureType],
) -> None:
    assert (
        addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=pl.DataFrame(
                [
                    {"sandbox_message_index": 0, "content": "Hello there", "num": 2},
                ]
            ),
        )
        == 0
    )
    assert (
        addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    assert (
        addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 0},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        == 0
    )
    assert (
        0
        < addition_similarity(
            snapshot=snapshot,
            target_dataframe=pl.DataFrame(
                [
                    {"content": "Hi there", "num": 1},
                ]
            ),
            column_similarities=column_similarities,
            reference_snapshot=reference_snapshot,
        )
        < 1
    )


@pytest.fixture
def tool_trace_snapshot() -> pl.DataFrame:
    """Testing snapshot containing a tool_trace.

    Returns:
        Testing snapshot containing a tool_trace.
    """
    return pl.DataFrame(
        [
            {
                "tool_trace": [
                    json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                                "year": 2024,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "tool_name": "search_lat_lon",
                            "arguments": {
                                "latitude": 37.334606,
                                "longitude": -122.009102,
                            },
                        },
                        ensure_ascii=False,
                    ),
                ]
            }
        ],
        schema={
            "tool_trace": pl.List(pl.String),
        },
    )


def test_column_tool_trace_exact_match_similarity(
    tool_trace_snapshot: pl.DataFrame,
) -> None:
    # Matching against 1 golden trace. Only present arguments are checked
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_stock",
                    "arguments": {
                        "query": "Apple",
                    },
                }
            ),
        )["similarity"][0]
        == 0
    )
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_holiday",
                    "arguments": {
                        "holiday_name": "Christmas Day",
                        "year": None,
                    },
                }
            ),
        )["similarity"][0]
        == 0
    )
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_holiday",
                    "arguments": {
                        "holiday_name": "Christmas Day",
                        "year": 2024,
                    },
                }
            ),
        )["similarity"][0]
        == 1
    )
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_holiday",
                    "arguments": {
                        "year": 2024,
                    },
                }
            ),
        )["similarity"][0]
        == 1
    )
    # Test multiple golden traces. Should match if one of them matches
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                [
                    {
                        "tool_name": "search_holiday",
                        "arguments": {
                            "holiday_name": "Christmas Day",
                            "year": None,
                        },
                    },
                    {
                        "tool_name": "search_holiday",
                        "arguments": {
                            "holiday_name": "Christmas Day",
                            "year": 2024,
                        },
                    },
                ]
            ),
        )["similarity"][0]
        == 1
    )
    # Exact match float
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_lat_lon",
                    "arguments": {
                        "latitude": 37.34,
                        "longitude": -122.01,
                    },
                },
            ),
        )["similarity"][0]
        == 0
    )
    # atol in range
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_lat_lon",
                    "arguments": {
                        "latitude": 37.34,
                        "longitude": -122.01,
                    },
                },
            ),
            atol_dict={"latitude": 0.2, "longitude": 0.2},
        )["similarity"][0]
        == 1
    )
    # atol out of range
    assert (
        column_tool_trace_exact_match_similarity(
            dataframe=tool_trace_snapshot,
            column_name="tool_trace",
            value=json.dumps(
                {
                    "tool_name": "search_lat_lon",
                    "arguments": {
                        "latitude": 37.34,
                        "longitude": -122.01,
                    },
                },
            ),
            atol_dict={"latitude": 0.002, "longitude": 0.002},
        )["similarity"][0]
        == 0
    )


@pytest.fixture
def tool_trace_dependant_similarity_kwargs() -> dict[str, Any]:
    """Testing kwargs for tool_trace_dependant_similarity.

    Returns:
        Testing kwargs for tool_trace_dependant_similarity.
    """
    return {
        "snapshot": pl.DataFrame(
            [
                {
                    "tool_trace": [
                        json.dumps(
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                    "year": 2024,
                                },
                            },
                            ensure_ascii=False,
                        ),
                    ]
                }
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        ),
        "reference_snapshot": pl.DataFrame(
            [
                {
                    "tool_trace": [
                        json.dumps(
                            {"results": 2024},
                            ensure_ascii=False,
                        ),
                    ]
                }
            ],
            schema=ExecutionContext.dbs_schemas[DatabaseNamespace.SANDBOX],
        ),
        "column_similarities": {"tool_trace": column_tool_trace_exact_match_similarity},
        "fill_to": "tool_trace",
    }


def test_tool_trace_dependant_similarity(
    tool_trace_dependant_similarity_kwargs: dict[str, Any],
) -> None:
    # Test extracting 1 argument
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"]}],
        )
        == 1
    )
    # Test extraction error
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"day": x["results"]}],
        )
        == 0
    )
    # Test incorrect extracted argument
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"] + 1}],
        )
        == 0
    )
    # Test prefers provided argument 1 argument
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        {
                            "tool_name": "search_holiday",
                            "arguments": {
                                "holiday_name": "Christmas Day",
                                "year": 2024,
                            },
                        },
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"] + 1}],
        )
        == 1
    )
    # Test multiple traces
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        [
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                    "year": None,
                                },
                            },
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                },
                            },
                        ],
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"]}],
        )
        == 1
    )
    # Test multiple traces, preferring provided value
    assert (
        tool_trace_dependant_similarity(
            **tool_trace_dependant_similarity_kwargs,
            target_dataframe=pl.DataFrame(
                {
                    "tool_trace": json.dumps(
                        [
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                    "year": 2024,
                                },
                            },
                            {
                                "tool_name": "search_holiday",
                                "arguments": {
                                    "holiday_name": "Christmas Day",
                                },
                            },
                        ],
                        ensure_ascii=False,
                    ),
                }
            ),
            extractor=lambda x: [{"year": x["results"] + 1}],
        )
        == 1
    )
