import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

AGENT_META = {
    "order_inquiry": {"label": "OrderAgent", "icon": "📦", "color": "#2563eb"},
    "retention_risk": {"label": "RetentionAgent", "icon": "💝", "color": "#dc2626"},
    "policy_question": {"label": "PolicyAgent (RAG)", "icon": "📖", "color": "#059669"},
    "escalation": {"label": "EscalationAgent", "icon": "🆘", "color": "#d97706"},
}

EXAMPLE_PROMPTS = [
    ("📦 Check an order", "Where is my order ORD-000001?"),
    ("📖 Ask about policy", "What is your policy on refunds for damaged items?"),
    ("🚫 Test refusal", "Can I pay for my order in Bitcoin?"),
    ("💝 Retention scenario", "I'm customer CUST-00093 and I'm thinking about cancelling, this keeps happening"),
    ("🆘 Escalation", "I want to speak to a manager, I've asked about this three times already"),
]

st.set_page_config(page_title="Customer Intelligence Agent", page_icon="🤖", layout="wide")

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
    .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

st.title("🤖 Multi-Agent Customer Intelligence & Decision Engine")
st.caption(
    "A router classifies each message, then hands off to a specialized agent -- "
    "each backed by real Postgres reads/writes or hybrid RAG, not a single prompt."
)

with st.sidebar:
    st.subheader("Architecture")
    st.markdown(
        "**Message** → Router (intent classification)\n\n"
        "→ 📦 **OrderAgent** -- checks/creates orders in Postgres\n\n"
        "→ 💝 **RetentionAgent** -- reads real churn risk, sizes discount\n\n"
        "→ 📖 **PolicyAgent** -- BM25 + dense + reranked RAG, cites or refuses\n\n"
        "→ 🆘 **EscalationAgent** -- logs a ticket for human follow-up"
    )
    st.divider()
    st.subheader("Try an example")
    for label, prompt in EXAMPLE_PROMPTS:
        if st.button(label, use_container_width=True, key=f"ex-{label}"):
            st.session_state.pending_prompt = prompt
    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Thread: `{st.session_state.thread_id}`")


def render_message(role, content, intent):
    icon = "🧑" if role == "user" else AGENT_META.get(intent, {}).get("icon", "🤖")
    with st.chat_message(role, avatar=icon):
        st.markdown(content)
        if intent and role == "assistant":
            meta = AGENT_META.get(intent)
            if meta:
                st.markdown(
                    f'<span class="agent-badge" style="background:{meta["color"]}">'
                    f'{meta["icon"]} {meta["label"]}</span>',
                    unsafe_allow_html=True,
                )


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("intent"))


def send_message(user_text):
    st.session_state.messages.append({"role": "user", "content": user_text})
    render_message("user", user_text, None)

    with st.spinner("Routing to the right agent..."):
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"thread_id": st.session_state.thread_id, "message": user_text},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            answer, intent = data["response"], data.get("intent")
        except Exception as e:
            answer, intent = f"⚠️ Error reaching backend: {e}", None

    render_message("assistant", answer, intent)
    st.session_state.messages.append({"role": "assistant", "content": answer, "intent": intent})


if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    send_message(prompt)

if prompt := st.chat_input("Type a message..."):
    send_message(prompt)