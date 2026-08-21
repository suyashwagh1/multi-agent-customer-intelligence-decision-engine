import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

from app.graph.state import GraphState
from app.schemas.outputs import RouterOutput

_router_llm = ChatAnthropic(
    model="claude-sonnet-5",
    api_key=os.environ["ANTHROPIC_API_KEY"],
).with_structured_output(RouterOutput)

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a customer support system.
Classify the customer's message into exactly one category:

- order_inquiry: questions about order status, shipping, delivery, or wanting to
  return/exchange an item.
- retention_risk: customer expresses intent to cancel, switch to a competitor,
  or shows frustration threatening the relationship.
- policy_question: questions about refund policy, return windows, shipping SLAs,
  or other stated policies -- not about a specific order's status.
- escalation: customer explicitly asks for a manager/human, or describes a
  repeated unresolved issue across multiple prior contacts.

If a message could fit more than one category, pick the most urgent/specific one.
If confidence is low, prefer escalation over guessing."""


def router_node(state: GraphState):
    """Classifies the latest human message into one of four intents."""
    last_message = state["messages"][-1]
    result: RouterOutput = _router_llm.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": last_message.content},
        ]
    )
    return {"intent": result.intent}


def route_by_intent(state: GraphState) -> str:
    """Conditional edge function -- maps classified intent to the next node."""
    intent = state.get("intent", "escalation")
    mapping = {
        "order_inquiry": "order_agent",
        "retention_risk": "retention_agent",
        "policy_question": "policy_agent",
        "escalation": "escalation_agent",
    }
    return mapping.get(intent, "escalation_agent")