# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import pytest

from tool_sandbox.cli.utils import resolve_scenarios
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios


def test_getting_all_scenarios() -> None:
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=None, preferred_tool_backend=ToolBackend.DEFAULT
    )
    assert set(
        named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT).keys()
    ) == set(name_to_scenario.keys())


def test_getting_no_scenarios() -> None:
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=[], preferred_tool_backend=ToolBackend.DEFAULT
    )
    assert 0 == len(name_to_scenario)


def test_getting_desired_scenarios() -> None:
    # Pick the first N names from all available scenarios to ensure that this test is
    # using existing scenario names.
    desired_scenario_names = list(
        named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT).keys()
    )[:5]
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=desired_scenario_names,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )
    assert set(desired_scenario_names) == set(name_to_scenario.keys())


def test_getting_non_existent_scenarios() -> None:
    all_scenario_names = set(
        named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT).keys()
    )
    non_existent_scenario = "this scenario does not exist"
    assert non_existent_scenario not in all_scenario_names

    with pytest.raises(KeyError, match="desired scenarios do not exist"):
        resolve_scenarios(
            desired_scenario_names=[non_existent_scenario],
            preferred_tool_backend=ToolBackend.DEFAULT,
        )
