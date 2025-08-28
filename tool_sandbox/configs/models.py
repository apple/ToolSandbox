"""Configuration for models in PAR package."""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# import model aliases
from tool_sandbox.models.utils import MODEL_ALIASES


class PromptConfig(BaseModel):
    """Configuration for the prompts used in the language model.

    Create a new instance of this class for each prompt set.
    """

    system_prompt: str = Field(..., description="The system prompt for the language model.")
    task_prompt: str = Field(..., description="The initial user message decribing the task for the language model.")
    feedback_prompt: str | None = Field(
        default=None,
        description="The feedback prompt for the language model. This is used when the generated hypothesis does not meet the criteria.",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> PromptConfig:
        """Load the prompt config from a YAML file.

        Args:
            yaml_path (str): The path to the YAML file.

        Returns:
            PromptConfig: The prompt config.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# destination: src/config.py
class APIModelConfig(BaseModel):
    """Configuration for the API model."""

    model_name: str = Field(
        default="gpt4o",
        description="Full litellm model name of the model. See https://docs.litellm.ai/docs/providers",
    )
    temperature: float = Field(default=0.7, description="The temperature for the model.")
    top_p: float = Field(default=1.0, description="The top-p value for the model.")
    max_tokens: int = Field(default=1000, description="The maximum number of tokens to generate.")
    host_url: str | None = Field(
        default=None, description="The base URL for the model. Used for some custom model providers."
    )
    completion_kwargs: dict[str, Any] = Field(
        default={}, description="additional kwargs for the litellm.completion call."
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate the model name.

        Args:
            v: The model name to validate.

        Returns:
            The validated model name.
        """
        if v in MODEL_ALIASES:
            return MODEL_ALIASES[v]
        return v


# destination: src/models.py
class APIStats(BaseModel):
    """Statistics for the API model."""

    cost: float = 0.0
    tokens_received: int = 0
    tokens_sent: int = 0
    api_calls: int = 0

    def __add__(self, other: APIStats) -> APIStats:
        """Add two APIStats objects together.

        Args:
            other: The other APIStats object to add to the current object.

        Returns:
            A new APIStats object with the sum of the two objects.

        Raises:
            TypeError: If other is not an APIStats object.
        """
        if not isinstance(other, APIStats):
            raise TypeError("APIStats objects can only be added to other APIStats objects")

        return APIStats(**{field: getattr(self, field) + getattr(other, field) for field in self.model_fields})

    def __replace__(self, other: APIStats) -> APIStats:
        """Replace the current object with the other object.

        Args:
            other: The other APIStats object to replace the current object with.

        Returns:
            APIStats: A new APIStats object with valued from other

        Raises:
            TypeError: If other is not an APIStats object.
        """
        if not isinstance(other, APIStats):
            raise TypeError("APIStats objects can only be replaced with other APIStats objects")

        return APIStats(**{field: getattr(other, field) for field in self.model_fields})
