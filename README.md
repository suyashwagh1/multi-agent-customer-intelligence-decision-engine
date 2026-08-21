# Multi-Agent Customer Intelligence & Decision Engine

A router-based multi-agent customer support system built on LangGraph and Claude, backed by a real PostgreSQL database. Instead of a single prompt answering everything, an intent classifier routes each customer message to one of four specialized agents -- each restricted to its own tools, each capable of taking real, persisted actions (order lookups, return requests, retention discounts, escalation tickets) rather than just generating text. A hybrid retrieval pipeline (BM25 + dense embeddings + cross-encoder reranking) grounds policy answers in actual source documents, and refuses to answer when no confident match exists rather than hallucinating.

LINK - https://multi-agent-customer-intelligence-decision-engine-exrivq3futtm.streamlit.app/

## Overview

Customer support systems generate two kinds of data: transactional (orders, returns) and conversational (support tickets, complaints). Making good decisions from this data -- who's at risk of churning, what a policy actually says, when to escalate to a human -- requires combining structured customer intelligence with agentic reasoning over unstructured language.

This project builds that combination end to end: a synthetic but behaviorally realistic dataset, a derived customer risk-scoring layer, a multi-agent orchestration system with real database actions, and a retrieval-augmented policy layer with measured evaluation results.

## Key Capabilities

- **Intent-based routing**: a structured-output classifier (Pydantic schema, not free-text parsing) routes every message to `order_inquiry`, `retention_risk`, `policy_question`, or `escalation`.
- **Four specialist agents**, each with its own system prompt and tool set:
  - `OrderAgent` -- checks order status, files return requests (real Postgres writes)
  - `RetentionAgent` -- reads a customer's actual churn-risk profile before offering a discount, capped at 20% per policy
  - `PolicyAgent` -- hybrid RAG over policy documents, cites its source or explicitly refuses
  - `EscalationAgent` -- logs a ticket for human follow-up
- **Real actions, not text**: return requests, retention discounts, and escalations are actual `INSERT`s into PostgreSQL, verified directly via `psql` during development, not just conversational claims.
- **Hybrid retrieval**: BM25 (keyword) + dense embeddings (semantic) combined via Reciprocal Rank Fusion, then reranked with a cross-encoder.
- **Grounded refusal**: when retrieval has no confident match, the PolicyAgent says so and escalates instead of guessing.
- **Derived customer intelligence**: churn risk, CLV estimate, and customer segment are computed from raw orders/tickets by an explicit, explainable weighted formula -- not loaded from a precomputed source.
- **Persistent state**: LangGraph's `PostgresSaver` checkpointer, so conversations survive a process restart.
- **HTTP API + interactive UI**: FastAPI backend, Streamlit frontend showing which agent handled each message.
- **RAGAS-style evaluation**: faithfulness and context-precision scoring on the PolicyAgent, implemented directly against Claude as judge.

## Architecture

```mermaid
flowchart TD
    U[Customer message] --> R[Router<br/>Claude + structured output]
    R -->|order_inquiry| OA[OrderAgent]
    R -->|retention_risk| RA[RetentionAgent]
    R -->|policy_question| PA[PolicyAgent]
    R -->|escalation| EA[EscalationAgent]

    OA -->|check_order_status<br/>create_return_request| PG[(PostgreSQL)]
    RA -->|get_customer_profile<br/>apply_retention_discount| PG
    EA -->|log_escalation| PG

    PA --> HR[Hybrid Retriever]
    HR --> BM25[BM25 keyword search]
    HR --> DENSE[Dense embedding search<br/>all-MiniLM-L6-v2]
    BM25 --> FUSE[Reciprocal Rank Fusion]
    DENSE --> FUSE
    FUSE --> RERANK[Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2]
    RERANK -->|confident match| PA
    RERANK -->|no confident match| EA
    PA -->|cites source| PGV[(pgvector: policy_chunks)]
```

Each agent's tool-calling loop is self-contained: `agent -> tools -> agent -> END`. A tool-calling bug in one agent cannot leak into another agent's behavior, since they don't share a tool list.

## Data Layer

```
data/
├── raw/                 # source data -- not manually edited
│   ├── customers.csv     (1,000 rows)
│   ├── products.csv      (100 rows)
│   ├── orders.csv        (8,000 rows)
│   └── tickets.csv       (2,651 rows -- 493+ unique message templates
│                           with slot-filled variation, not canned strings)
├── processed/
│   └── customer_metrics.csv   # OUTPUT of scripts/compute_customer_metrics.py
└── policies/             # source documents for the RAG layer
    ├── refund_policy.md
    ├── return_policy.md
    ├── shipping_sla.md
    ├── retention_policy.md
    └── escalation_rules.md
```

`customer_metrics.csv` is never hand-authored or loaded as ground truth -- it is generated fresh by running `compute_customer_metrics.py` against the raw orders/tickets tables, and `seed_db.py` loads that output into Postgres. A separate script, `inject_at_risk_cohort.py`, deliberately shapes a realistic High-Value/At-Risk cohort by editing raw behavioral signals (return status, recent ticket volume) on a subset of high-spend customers -- it never writes a risk score directly; the score is always recomputed downstream by the pipeline.

**Current dataset statistics** (verified via `SELECT COUNT(*)` against the live database):

| Table | Rows |
|---|---:|
| customers | 1,000 |
| products | 100 |
| orders | 8,000 |
| tickets | 2,651 |
| customer_metrics | 1,000 |
| policy_chunks | 28 |

## Customer Intelligence

`churn_risk_score` is a **transparent weighted formula**, not a trained ML model:

```
churn_risk_score = 0.35 x recency_score
                  + 0.20 x return_score
                  + 0.25 x ticket_score
                  + 0.20 x sentiment_score
```

where each component is normalized to 0-1 from raw behavior (days since last order, return rate, support tickets in the last 90 days, share of negative-sentiment tickets). This was a deliberate choice: for an agentic orchestration project, an explainable heuristic is easier to defend under questioning than a black-box model, and it keeps the project's focus on the agent architecture rather than a separate ML training pipeline.

Current risk distribution:

| Segment | Count |
|---|---:|
| Low risk | 520 |
| Medium risk | 441 |
| High risk | 39 |
| **High-Value/At-Risk** (the interesting case for RetentionAgent) | 11 |

## Agent Workflows

### OrderAgent
- **Tools**: `check_order_status(order_id)`, `create_return_request(order_id, customer_id, reason)`
- **Behavior**: reads live order/product data via SQL join; creates a `return_request` ticket (with JSONB details) only after confirming the order exists and belongs to the stated customer.

### RetentionAgent
- **Tools**: `get_customer_profile(customer_id)`, `apply_retention_discount(customer_id, discount_pct, justification)`
- **Behavior**: system prompt requires calling `get_customer_profile` before offering anything. Discount sizing is left to the model's judgment against policy guidance (max 20%, reserved for genuinely high-value/high-risk customers) rather than a fixed rule -- verified in testing to correctly size a smaller discount for a lower-value customer and the full 20% for a verified high-value/high-risk one (`CUST-00093`: $4,468.97 spend, 0.689 churn risk).

### PolicyAgent
- **Tool**: `retrieve_policy(query)` -- wraps the hybrid retriever
- **Behavior**: system prompt explicitly forbids answering from general knowledge. Must cite the retrieved section or relay a `NO_CONFIDENT_MATCH` signal and defer to escalation.

### EscalationAgent
- **Tool**: `log_escalation(customer_id, reason)`
- **Behavior**: terminal handoff node. Also currently receives any message the router itself is unsure how to classify.

## Policy RAG

1. **Chunking** (`app/rag/chunker.py`): each policy `.md` file is split by `##` section headers into topically coherent chunks (28 total across 5 documents).
2. **Embedding** (`app/rag/embeddings.py`): `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional, run locally -- no embedding API cost.
3. **Storage**: chunks and embeddings are stored in `policy_chunks` (Postgres + `pgvector`).
4. **Retrieval** (`app/rag/hybrid_retriever.py`), fully hybrid, not embeddings-only:
   - BM25 keyword ranking (`rank_bm25`, custom tokenizer stripping punctuation)
   - Dense cosine-similarity ranking over the stored embeddings
   - Reciprocal Rank Fusion to combine both rankings
   - Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the fused top candidates, with a small relevance boost when a candidate's section title shares words with the query
5. **Refusal**: if the top reranked score falls below a confidence threshold, retrieval returns no result and the PolicyAgent is instructed to escalate rather than answer.

## Evaluation

A RAGAS-style evaluation (`eval/run_ragas.py`) scores the PolicyAgent's faithfulness (does the answer stay within the retrieved context?) and context precision (was the retrieved context actually relevant?), using Claude as judge. This is implemented directly rather than via the `ragas` package, which had a broken, unrelated dependency chain in the development environment at the time of writing.

**Development note, included for honesty rather than omitted:** the first evaluation run surfaced a real retrieval bug -- the query "What is your policy on refunds for damaged items?" retrieved *Return Policy -- Category Exclusions* (a short cross-reference sentence that happened to contain several query words) instead of the substantively correct *Refund Policy -- Damaged or Defective Items* section. Root cause: BM25's default length normalization penalized the longer, more relevant chunk, and the cross-encoder reranker mildly shared the same bias. Two targeted fixes resolved it: reducing BM25's length-normalization parameter (`b=0.4`), and adding a small reranking boost when a candidate's section title shares words with the query.

| Metric | Before fix | After fix |
|---|---:|---:|
| Average faithfulness | 1.00 | 1.00 |
| Average context precision | 0.78 | 0.98 |
| Correct refusal (Bitcoin question) | 1/1 | 1/1 |

Faithfulness was 1.00 even before the fix -- the agent never invented facts, it faithfully summarized whatever context it was given, even when that context was the wrong section. This is exactly why RAGAS scores faithfulness and context precision separately: high faithfulness with low context precision isolates the problem to retrieval, not generation.

The evaluation set (`eval/test_conversations.json`) currently contains 12 hand-written cases across policy grounding, refusal, ambiguous intent, retention sizing at both high and low customer value, escalation, and a multi-turn cancel-then-recant scenario. Only the 4 `policy_*` cases are currently scored by the RAGAS harness; the rest are used for manual/regression testing of routing and tool behavior.

## Project Structure

```
multi-agent-customer-intelligence-decision-engine/
├── app/
│   ├── api.py                  # FastAPI app (/chat, /health)
│   ├── main.py                  # CLI test harness
│   ├── db/
│   │   └── session.py            # shared Postgres connection string
│   ├── graph/
│   │   ├── state.py               # GraphState (messages, intent)
│   │   ├── router.py              # intent classifier + routing function
│   │   └── build_graph.py         # full graph assembly, all 4 agents
│   ├── tools/
│   │   ├── order_tools.py
│   │   ├── return_tools.py
│   │   ├── retention_tools.py
│   │   ├── policy_tools.py
│   │   └── escalation_tools.py
│   ├── rag/
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── ingest.py
│   │   └── hybrid_retriever.py
│   └── schemas/
│       └── outputs.py             # RouterOutput (Pydantic)
├── data/
│   ├── raw/
│   ├── processed/
│   └── policies/
├── db/
│   └── schema.sql
├── scripts/
│   ├── generate_tickets.py
│   ├── inject_at_risk_cohort.py
│   ├── compute_customer_metrics.py
│   └── seed_db.py
├── eval/
│   ├── test_conversations.json
│   └── run_ragas.py
├── streamlit_app.py
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

```powershell
# 1. Clone
git clone https://github.com/suyashwagh1/multi-agent-customer-intelligence-decision-engine.git
cd multi-agent-customer-intelligence-decision-engine

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env
# then edit .env with your real ANTHROPIC_API_KEY

# 4. Start PostgreSQL (schema.sql applies automatically on first boot)
docker compose up -d

# 5. Seed the database
python scripts/generate_tickets.py
python scripts/inject_at_risk_cohort.py
python scripts/compute_customer_metrics.py
python scripts/seed_db.py

# 6. Ingest policy documents into pgvector
python -m app.rag.ingest

# 7a. Run via CLI
python -m app.main

# 7b. OR run via API + UI (two terminals)
python -m uvicorn app.api:app --reload --port 8000
python -m streamlit run streamlit_app.py

# 8. Run evaluation
python -m eval.run_ragas
```

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | Postgres connection string, e.g. `postgresql://cia_user:cia_password@localhost:5432/customer_intelligence` |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (`sk-ant-...`) -- see `.env.example` |

## Example Usage

All examples below were tested against the running system during development.

**Order lookup:**
> "Where is my order ORD-000002?"
> → *"Your order ORD-000002 (Product 20, $825.62, placed 2025-09-30) has shipped and is on its way!"*

**Retention scenario (high-value/high-risk):**
> "I'm customer CUST-00093 and I'm thinking about cancelling, this keeps happening"
> → RetentionAgent calls `get_customer_profile`, sees $4,468.97 spend / 0.689 churn risk / 6 tickets in 90 days, applies the full 20% discount with a justification tied to those numbers.

**Policy question (grounded):**
> "What is your policy on refunds for damaged items?"
> → Cites *Refund Policy -- Damaged or Defective Items*: full refund within 45 days of delivery, photo evidence may be requested, no return shipment required if damage is visually confirmed.

**Policy question (no match -- refusal):**
> "Can I pay for my order in Bitcoin?"
> → *"I don't have a confident answer on whether Bitcoin is accepted... I'll escalate this to a human agent."*

**Escalation:**
> "I want to speak to a manager, I've asked about this three times already"
> → Logs an escalation ticket and hands off.

## Database

Core tables: `customers`, `products`, `orders`, `tickets`, `customer_metrics`, `policy_chunks`. `orders` and `tickets` both reference `customers.customer_id`; `orders` also references `products.product_id`. `customer_metrics` is keyed 1:1 to `customers` and is fully derived, never hand-edited. `policy_chunks` holds RAG source text plus a `vector(384)` embedding column. LangGraph's `PostgresSaver` checkpointer additionally creates and manages its own tables for conversation state -- not defined in `schema.sql`, created automatically on first run.

## Evaluation & Limitations

- The RAGAS-style harness currently scores only 4 policy cases -- a real but small sample; the retrieval fix described above was validated on this same small set, not a held-out one.
- All data (customers, orders, tickets) is synthetic, generated with `Faker`/scripted distributions plus a deliberately engineered at-risk cohort -- not real customer data.
- `churn_risk_score` is a hand-weighted heuristic, not a trained/validated predictive model.
- The router, when uncertain, currently sends `policy_question` misclassifications nowhere special -- all four categories map directly to their agent, there is no confidence-based fallback routing yet beyond the retrieval layer's own refusal behavior.
- No load testing or concurrency testing has been performed.

## Future Work

- Expand the evaluation set beyond 12 hand-written cases, and score routing accuracy (not just PolicyAgent retrieval) with RAGAS-style metrics.
- Confidence-based router fallback (route low-confidence classifications to escalation).
- Live deployment (hosted Postgres, hosted API, hosted UI).
- Query rewriting or multi-query retrieval to further improve context precision on differently-phrased questions.

## Security

- Real secrets live only in `.env`, which is excluded via `.gitignore` and never committed.
- `.env.example` contains placeholders only.
- Database credentials in `docker-compose.yml` are local development defaults, not production secrets.


## Author
Suyash Wagh
