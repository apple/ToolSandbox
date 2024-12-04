
 

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.utils import register_as_tool
from tool_sandbox.common.validators import typechecked, validate_type


@register_as_tool(visible_to=(RoleType.AGENT,))
@typechecked
def chat(message: str) -> None:
    """
    Always use this tool at the end of the conversation (or in the middle if necessary) to
    communicate to the user in natural language to address something has been accomplished
    or if you need additional information.

    You can also use this tool if the user query is NOT answerable by the other tools. This
    can act as an out of domain function.

    You can also use this tool to ask the user for more information if there is insufficient
    information present in their current query.

    Args:
        message: a clear message that the user will read.

    Returns:
        None
    """
    validate_type(message, "message", str)
    return None
