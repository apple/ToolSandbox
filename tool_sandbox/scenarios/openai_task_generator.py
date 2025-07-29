from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from requests.exceptions import HTTPError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from tool_sandbox.common.execution_context import ExecutionContext
from tool_sandbox.scenarios.scenario_generator import TaskGenerator


class OpenAITaskGenerator(TaskGenerator):
    """Task generator that uses OpenAI API."""

    def __init__(self, execution_context: ExecutionContext, model_name: str = "gpt-4o", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the OpenAI task generator.

        Args:
            execution_context: The execution context
            model_name: The name of the model to use
            **kwargs: Additional arguments to pass to the TaskGenerator constructor
        """
        super().__init__(execution_context, **kwargs)
        self.model_name = model_name
        self.openai_client = OpenAI(base_url="https://api.openai.com/v1")

    def model_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a task using the OpenAI API.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt

        Returns:
            The generated task string.
        """
        if not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})

        self.messages.append({"role": "user", "content": user_prompt})

        response = self._call_openai_api(self.messages)

        response_content: str | None = response.choices[0].message.content
        if response_content is None:
            raise RuntimeError("OpenAI API return empty response.")

        self.messages.append({"role": "assistant", "content": response_content})

        return response_content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=40),
        retry=retry_if_exception_type(HTTPError),
    )
    def _call_openai_api(self, messages: list[dict[str, str]]) -> ChatCompletion:
        if self.model_name in ["o3-mini", "o4-mini", "o3"]:
            response = self.openai_client.chat.completions.create(
                model=self.model_name, messages=cast("list[ChatCompletionMessageParam]", messages)
            )
        else:
            response = self.openai_client.chat.completions.create(
                model=self.model_name, messages=cast("list[ChatCompletionMessageParam]", messages), temperature=0.8
            )
        return response
