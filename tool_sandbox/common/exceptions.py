"""Custom exceptions for Proactive Goal Inference package."""

from __future__ import annotations


class CostLimitExceededError(Exception):
    """Exception raised when the cost limit is exceeded."""

    pass


class FormatError(Exception):
    """Exception raised when the format of the model response is invalid."""

    pass
