"""Parser implementations for model outputs in goal inference tasks."""

import re
from typing import ClassVar

from tool_sandbox.models.utils import ParseFunction


class BoxedGoalInferenceParser(ParseFunction):
    """Parser for goal inference outputs in [[content]] format.

    Expects model outputs in the format:
    - [[wait]] - when the model needs more information
    - [[some goal description]] - when the model can infer the goal
    """

    _error_message: ClassVar[str] = (
        "Invalid format. Please respond with either [[wait]] if you need more information, "
        "or [[your goal prediction]] if you can infer the user's goal. "
        "Your response must be exactly in this format with double square brackets."
    )

    def __call__(self, model_response: str) -> dict[str, str]:
        """Parse the model response for goal inference.

        Args:
            model_response: The raw response from the model

        Returns:
            dict with keys:
            - "prediction_type": either "wait" or "goal"
            - "content": the content inside the brackets

        Raises:
            ValueError: If the response doesn't match the expected format
        """
        # Clean the response
        cleaned_response = model_response.strip()

        # Look for [[content]] pattern
        pattern = r"\[\[(.+?)\]\]"
        match = re.search(pattern, cleaned_response)

        if not match:
            raise ValueError(f"No bracketed content found. {self.format_error_template}")

        content = match.group(1).strip()

        if not content:
            raise ValueError(f"Empty bracketed content. {self.format_error_template}")

        # Determine if it's wait or goal prediction
        prediction_type = "wait" if content.lower() == "wait" else "goal"

        return {"prediction_type": prediction_type, "content": content}
