"""LiteLLM based model inference.

Ideally this will be used for all model inference needs, agent, user, proactive etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
)
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from tool_sandbox.configs.models import APIStats

if TYPE_CHECKING:
    import logging

    from tool_sandbox.configs.models import APIModelConfig


class LiteLLMModel:
    """A wrapper for the LiteLLM API. Inherit this class to make it a stateful model.

    Cost state management is included in this class, so the inherited classes can implement their own cost management or reuse this class.
    """

    def __init__(self, config: APIModelConfig, name: str, logger: logging.Logger) -> None:
        """Initialize the LiteLLM API Model.

        Args:
            config: The configuration for the LiteLLM API Model
            name: The name of the LiteLLM API Model
            logger: The logger for the LiteLLM API Model
        """
        self.name = name
        self.config = config
        self.stats = APIStats()
        self.logger = logger
        self._setup_client()

    def _setup_client(self) -> None:
        self.model_name = self.config.model_name
        self.model_max_input_tokens = litellm.model_cost.get(self.model_name, {}).get("max_input_tokens")
        self.model_max_output_tokens = litellm.model_cost.get(self.model_name, {}).get("max_output_tokens")
        self.lm_provider = litellm.model_cost.get(self.model_name, {}).get("litellm_provider")
        if self.lm_provider is None or self.config.host_url is not None:
            self.logger.warning(
                f"Using custom host URL: {self.config.host_url}. Cost management will not be available. Register the model with litellm to enable cost management. See https://docs.litellm.ai/docs/completion/token_usage#9-register_model"
            )

    def reset_stats(self, other: APIStats) -> None:
        """Reset or replace the current API Statistics.

        Args:
            other: The other APIStats object to replace the current object with.
        """
        self.stats = other

    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> None:
        """Update API statistics with new usage information.

        Args:
            input_tokens (int): Number of tokens in the prompt
            output_tokens (int): Number of tokens in the response
            cost (float): Cost of the API call. Defaults to 0.0

        Returns:
            float: The calculated cost of the API call
        """
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.cost += cost
        self.stats.api_calls += 1

    @retry(
        wait=wait_random_exponential(min=180, max=360),
        reraise=True,
        stop=stop_after_attempt(3),
        retry=retry_if_not_exception_type(
            (
                RuntimeError,
                NotFoundError,
                PermissionDeniedError,
                ContextWindowExceededError,
                APIError,
                AuthenticationError,
                BadRequestError,
            )
        ),
    )
    def query(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> litellm.types.utils.ModelResponse:
        """Query the model with messages and optional tools.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions in OpenAI format

        Returns:
            litellm.types.utils.ModelResponse: The complete response from the model
        """
        input_tokens = litellm.utils.token_counter(messages=messages, model=self.model_name)
        extra_args = {}
        if self.config.host_url:
            extra_args["api_base"] = self.config.host_url

        completion_kwargs = self.config.completion_kwargs.copy()
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens

        # Add tools if provided
        if tools is not None:
            completion_kwargs["tools"] = tools

        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=False,
                **completion_kwargs,
                **extra_args,
            )
        except Exception:
            self.logger.exception(f"Error querying {self.model_name} with tools")
            raise

        choices: litellm.types.utils.Choices = response.choices  # type: ignore
        output_content = choices[0].message.content or ""

        self.logger.debug(f"Input:\n{messages[-1]['content'] if messages else 'No messages'}")
        self.logger.debug(f"Response: {output_content}")

        # Calculate output tokens
        # For tool calls, we need to count tokens in the entire response
        if hasattr(choices[0].message, "tool_calls") and choices[0].message.tool_calls:
            # Include tool calls in token count
            tool_calls_text = str(choices[0].message.tool_calls)
            output_tokens = litellm.utils.token_counter(
                text=f"{output_content}{tool_calls_text}", model=self.model_name
            )
        else:
            output_tokens = litellm.utils.token_counter(text=output_content, model=self.model_name)

        # Update stats
        cost = litellm.cost_calculator.completion_cost(response)
        self.update_stats(input_tokens, output_tokens, cost)

        self.logger.debug(
            f"input_tokens={input_tokens:,}, output_tokens={output_tokens:,}, instance_cost={cost:.2f}, "
            f"total_tokens_sent={self.stats.tokens_sent:,}, total_tokens_received={self.stats.tokens_received:,}, total_cost={self.stats.cost:.2f}, total_api_calls={self.stats.api_calls:,}"
        )

        return response
