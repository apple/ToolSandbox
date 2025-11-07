# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Test for value validators."""

from typing import Optional, Union

import pytest

from tool_sandbox.common.validators import (
    typechecked,
    validate_currency_code,
    validate_range,
    validate_type,
    validate_type_range,
)


def test_validate_currency_code() -> None:
    with pytest.raises(ValueError):
        assert validate_currency_code("$")
    validate_currency_code("USD")


@typechecked
def pos_args_func(
    int_arg: int,
    str_arg: str,
    opt_float: Optional[float],
    opt_int_or_float: Optional[Union[int, float]],
) -> str:
    """Function to be type checked."""
    return "valid"


@typechecked
def kwonly_args_func(*, int_arg: int, opt_str_arg: Optional[str] = None) -> int:
    """Keyword-only function to be type checked."""
    return 1


def test_typechecked_call() -> None:
    """Test `typechecked` decorated function calls."""
    assert "valid" == pos_args_func(1, "str", None, None)
    assert "valid" == pos_args_func(1, "str", 1.0, 1)
    assert "valid" == pos_args_func(1, "str", opt_float=None, opt_int_or_float=None)
    assert 1 == kwonly_args_func(int_arg=1)
    assert 1 == kwonly_args_func(int_arg=1, opt_str_arg="foo")
    with pytest.raises(TypeError):
        pos_args_func(1.001, "str", opt_float=None, opt_int_or_float=None)
    with pytest.raises(TypeError):
        pos_args_func(1, str_arg="str", opt_float=1, opt_int_or_float="fail")
    with pytest.raises(TypeError):
        pos_args_func(1, "str", 1.0, opt_int_or_float="fail")
    with pytest.raises(TypeError):
        kwonly_args_func(int_arg=1.01, opt_str_arg="foo")
    with pytest.raises(TypeError):
        kwonly_args_func(int_arg=1, opt_str_arg=2.0)
    with pytest.raises(TypeError):
        kwonly_args_func(int_arg="bar", opt_str_arg=None)


def test_validate_type() -> None:
    """Test that types are correctly validated."""
    validate_type("foo", "string_val", str)
    validate_type("foo", "string_val", Union[str, int])
    validate_type("foo", "string_val", Optional[Union[str, int]])
    validate_type(None, "none", Optional[str])
    validate_type(None, "none", Optional[Union[int, str]])
    validate_type(1, "int_val", Optional[Union[int, str]])

    validate_type(1, "int_val", int)
    validate_type_range(0, "int_val", int, min_val=0, max_val=0)
    validate_type_range(0, "int_val", int, min_val=0)
    validate_type_range(0, "int_val", int, max_val=0)
    validate_type_range(0, "int_val", int, min_val=-1, max_val=1)

    # Booleans are a subclass of integers and thus a valid input to a function accepting
    # integers.
    validate_type(True, "int_val", int)
    validate_type(False, "int_val", int)
    with pytest.raises(TypeError):
        validate_type(0, "int_val", bool)
    with pytest.raises(TypeError):
        validate_type(1, "int_val", bool)

    validate_type(9.9, "float_val", float)
    # An `int` is acceptable for a function argument typed as `float`.
    validate_type(9, "int_val", float)
    validate_type_range(0.0, "float_val", float, min_val=0.0, max_val=0.0)
    validate_type_range(0.0, "float_val", float, min_val=0.0)
    validate_type_range(0.0, "float_val", float, max_val=0.0)
    validate_type_range(0.0, "float_val", float, min_val=-1.0, max_val=1.0)

    with pytest.raises(TypeError):
        validate_type("foo", "string_val", int)
    with pytest.raises(TypeError):
        validate_type("foo", "string_val", Union[int, float])
    with pytest.raises(TypeError):
        validate_type(1.0, "float_val", str)
    with pytest.raises(TypeError):
        validate_type(1, "int_val", str)
    with pytest.raises(TypeError):
        validate_type(1.1, "float_val", int)
    with pytest.raises(TypeError):
        validate_type(1.0, "float_val", int)
    with pytest.raises(TypeError):
        validate_type(1, "int_val", Union[str, bool])
    with pytest.raises(TypeError):
        validate_type(None, "none", Union[str, float, int])

    with pytest.raises(ValueError):
        validate_range(100.0, "too_large_float_val", min_val=0.0, max_val=99.99)
    with pytest.raises(ValueError):
        validate_range(100.0, "too_small_float_val", min_val=100.0001, max_val=1000.0)
