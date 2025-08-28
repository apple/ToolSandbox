"""Utility functions for models module in Proactive Goal Inference package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from tool_sandbox.common.registry import RegistryMixin


class ParseFunction(RegistryMixin["ParseFunction"], ABC):
    """Abstract base class for parsing model outputs.

    Defines the interface for all parser implementations and provides registry-based instantiation.

    Args:
        _error_message (str | None): Template for error message to be used when parsing fails.
        _registry (dict[str, Type["ParserFunction"]): Registry of parser implementations
    """

    _error_message: ClassVar[str | None] = None
    _registry: ClassVar[dict[str, type[ParseFunction]]] = {}

    @abstractmethod
    def __call__(self, model_response: str) -> dict[str, str]:
        """Parse the model response and return the hypothesis.

        Args:
            model_response (str): The response from the model.

        Returns:
            Any: Any type of parsed output
        """
        raise NotImplementedError

    @property
    def format_error_template(self) -> str:
        """Get the error message template for format errors.

        Returns:
            str: The error message template

        Raises:
            NotImplementedError: If the subclass does not define _error_message
        """
        if self._error_message is None:
            raise NotImplementedError("Subclass must define _error_message")
        return self._error_message


MODEL_ALIASES = {
    "gpt4o": "gpt-4o",
    "bedrock-claude-3.5-sonnet": "bedrock/arn:aws:bedrock:us-west-2:288380904485:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    # Add more as needed
}
