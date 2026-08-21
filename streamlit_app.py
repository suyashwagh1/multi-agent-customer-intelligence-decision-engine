import os
import uuid

import streamlit as st

st.set_page_config(page_title="Customer Intelligence Agent", layout="wide")

try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    if "DATABASE_URL" in st.secrets:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except Exception:
    pass

from langchain_core.messages import HumanMessage

from app.graph.build_graph import build_graph

AGENT_META = {
    "order_inquiry": {"label": "OrderAgent", "color": "#2563eb"},
    "retention_risk": {"label": "RetentionAgent", "color": "#dc2626"},
    "policy_question": {"label": "PolicyAgent (RAG)", "color": "#059669"},
    "escalation": {"label": "EscalationAgent", "color": "#d97706"},
}

EXAMPLE_PROMPTS = [
    ("Check an order", "Where is my order ORD-000001?"),
    ("Ask about policy", "What is your policy on refunds for damaged items?"),
    ("Test refusal", "Can I pay for my order in Bitcoin?"),
    ("Retention scenario", "I'm customer CUST-00093 and I'm thinking about cancelling, this keeps happening"),
    ("Escalation", "I want to speak to a manager, I've asked about this three times already"),
]

st.markdown(
    """
    <style>
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    graph, checkpointer_cm = build_graph()
    return graph, checkpointer_cm


graph, _checkpointer_cm = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

st.title("Multi-Agent Customer Intelligence & Decision Engine")
st.caption(
    "A router classifies each message, then hands off to a specialized agent -- "
    "each backed by real Postgres reads/writes or hybrid RAG, not a single prompt."
)

with st.sidebar:
    st.subheader("Architecture")
    st.markdown(
        "**Message** -> Router (intent classification)\n\n"
        "-> **OrderAgent** -- checks/creates orders in Postgres\n\n"
        "-> **RetentionAgent** -- reads real churn risk, sizes discount\n\n"
        "-> **PolicyAgent** -- BM25 + dense + reranked RAG, cites or refuses\n\n"
        "-> **EscalationAgent** -- logs a ticket for human follow-up"
    )
    st.divider()
    st.subheader("Try an example")
    for label, prompt in EXAMPLE_PROMPTS:
        if st.button(label, use_container_width=True, key=f"ex-{label}"):
            st.session_state.pending_prompt = prompt
    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Thread: `{st.session_state.thread_id}`")


def render_message(role, content, intent):
    with st.chat_message(role):
        st.markdown(content)
        if intent and role == "assistant":
            meta = AGENT_META.get(intent)
            if meta:
                st.markdown(
                    f'<span class="agent-badge" style="background:{meta["color"]}">'
                    f'{meta["label"]}</span>',
                    unsafe_allow_html=True,
                )


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("intent"))


def run_agent(user_text):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    final_text = ""
    final_intent = None

    for event in graph.stream({"messages": [HumanMessage(content=user_text)]}, config):
        for value in event.values():
            if value.get("intent"):
                final_intent = value["intent"]
            messages = value.get("messages")
            if messages:
                message = messages[-1]
                if message.content and isinstance(message.content, str):
                    final_text = message.content

    return final_text, final_intent


def send_message(user_text):
    st.session_state.messages.append({"role": "user", "content": user_text})
    render_message("user", user_text, None)

    with st.spinner("Routing to the right agent..."):
        try:
            answer, intent = run_agent(user_text)
        except Exception as e:
            answer, intent = f"Error: {e}", None

    render_message("assistant", answer, intent)
    st.session_state.messages.append({"role": "assistant", "content": answer, "intent": intent})


if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    send_message(prompt)

if prompt := st.chat_input("Type a message..."):
    send_message(prompt)
