from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.graph.build_graph import build_graph

_graph_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph, pool = build_graph()
    _graph_state["graph"] = graph
    _graph_state["pool"] = pool
    yield
    _graph_state["pool"].close()


app = FastAPI(
    title="Multi-Agent Customer Intelligence & Decision Engine",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    intent: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    graph = _graph_state["graph"]
    config = {"configurable": {"thread_id": req.thread_id}}

    final_text = ""
    final_intent = None

    for event in graph.stream(
        {"messages": [HumanMessage(content=req.message)]},
        config,
    ):
        for value in event.values():
            if value.get("intent"):
                final_intent = value["intent"]
            messages = value.get("messages")
            if messages:
                message = messages[-1]
                if message.content and isinstance(message.content, str):
                    final_text = message.content

    return ChatResponse(thread_id=req.thread_id, response=final_text, intent=final_intent)


@app.get("/health")
async def health():
    return {"status": "ok"}