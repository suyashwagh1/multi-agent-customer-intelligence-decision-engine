from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]