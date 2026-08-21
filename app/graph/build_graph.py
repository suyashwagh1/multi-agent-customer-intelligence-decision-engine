import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg_pool import ConnectionPool

from app.db.session import DB_URL
from app.graph.router import route_by_intent, router_node
from app.graph.state import GraphState
from app.tools.escalation_tools import log_escalation
from app.tools.order_tools import check_order_status
from app.tools.policy_tools import retrieve_policy
from app.tools.retention_tools import apply_retention_discount, get_customer_profile
from app.tools.return_tools import create_return_request

load_dotenv()

API_KEY = os.environ["ANTHROPIC_API_KEY"]

order_tools = [check_order_status, create_return_request]
order_llm = ChatAnthropic(model="claude-sonnet-5", api_key=API_KEY).bind_tools(order_tools)

ORDER_SYSTEM_PROMPT = (
    "You are the OrderAgent for a customer support system. Help customers "
    "check order status and file return requests using your tools. Be "
    "concise and helpful."
)


def order_agent_node(state: GraphState):
    messages = [{"role": "system", "content": ORDER_SYSTEM_PROMPT}] + state["messages"]
    response = order_llm.invoke(messages)
    return {"messages": [response]}


retention_tools = [get_customer_profile, apply_retention_discount]
retention_llm = ChatAnthropic(model="claude-sonnet-5", api_key=API_KEY).bind_tools(retention_tools)

RETENTION_SYSTEM_PROMPT = (
    "You are the RetentionAgent. A customer is expressing frustration or "
    "intent to cancel. ALWAYS call get_customer_profile first to understand "
    "their value and risk level before offering anything. Size any discount "
    "based on their actual profile per policy (max 20%, and the max should "
    "be reserved for genuinely high-value, high-risk customers, not offered "
    "by default). Also address the underlying complaint, not just the discount."
)


def retention_agent_node(state: GraphState):
    messages = [{"role": "system", "content": RETENTION_SYSTEM_PROMPT}] + state["messages"]
    response = retention_llm.invoke(messages)
    return {"messages": [response]}


policy_tools = [retrieve_policy]
policy_llm = ChatAnthropic(model="claude-sonnet-5", api_key=API_KEY).bind_tools(policy_tools)

POLICY_SYSTEM_PROMPT = (
    "You are the PolicyAgent. Answer policy questions ONLY using "
    "retrieve_policy -- never from your own general knowledge. "
    "If the tool returns NO_CONFIDENT_MATCH, tell the customer you don't "
    "have a confident answer and that you're escalating this to a human "
    "agent -- do not guess or make up a policy. When you do have a good "
    "match, briefly cite which policy section it came from."
)


def policy_agent_node(state: GraphState):
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT}] + state["messages"]
    response = policy_llm.invoke(messages)
    return {"messages": [response]}


escalation_tools = [log_escalation]
escalation_llm = ChatAnthropic(model="claude-sonnet-5", api_key=API_KEY).bind_tools(escalation_tools)

ESCALATION_SYSTEM_PROMPT = (
    "You are the EscalationAgent. This conversation needs human follow-up -- "
    "either it was explicitly requested or it involves a repeated "
    "unresolved issue. Call log_escalation with a clear, brief reason, "
    "then let the customer know a human will follow up."
)


def escalation_agent_node(state: GraphState):
    messages = [{"role": "system", "content": ESCALATION_SYSTEM_PROMPT}] + state["messages"]
    response = escalation_llm.invoke(messages)
    return {"messages": [response]}


def _add_agent_with_tools(builder, name: str, node_fn, tools: list):
    tool_node_name = f"{name}_tools"
    builder.add_node(name, node_fn)
    builder.add_node(tool_node_name, ToolNode(tools))
    builder.add_conditional_edges(
        name,
        tools_condition,
        {"tools": tool_node_name, END: END},
    )
    builder.add_edge(tool_node_name, name)


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    _add_agent_with_tools(builder, "order_agent", order_agent_node, order_tools)
    _add_agent_with_tools(builder, "retention_agent", retention_agent_node, retention_tools)
    _add_agent_with_tools(builder, "policy_agent", policy_agent_node, policy_tools)
    _add_agent_with_tools(builder, "escalation_agent", escalation_agent_node, escalation_tools)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "order_agent": "order_agent",
            "retention_agent": "retention_agent",
            "policy_agent": "policy_agent",
            "escalation_agent": "escalation_agent",
        },
    )

    pool = ConnectionPool(
        conninfo=DB_URL,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
        check=ConnectionPool.check_connection,
        max_idle=300,
        reconnect_timeout=10,
    )
    pool.open(wait=True)

    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)
    return graph, pool