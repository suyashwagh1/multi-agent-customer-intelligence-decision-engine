"""
Customer Intelligence feature engineering pipeline.

Derives customer-level behavioral features and a churn-risk score from
RAW transactional/support data only (orders.csv, tickets.csv). This file
is an OUTPUT of the pipeline, not an input dataset — the whole point is
that the logic is ours and explainable, not a precomputed column we
loaded from a CSV.

churn_risk_score is a transparent weighted heuristic (not a trained
model) built from four signals:
  - recency      : days since last order, normalized
  - return_rate   : returns / orders
  - ticket_load   : support tickets in the last 90 days, normalized
  - sentiment_load: share of negative/very_negative tickets

This keeps the score interview-defensible: you can walk through exactly
why any given customer landed at a given risk level.
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime(2025, 9, 30)  # fixed "as-of" snapshot date, chosen to sit just
                                # after the latest ticket (2025-09-18) so the
                                # 90-day ticket window and recency scores are
                                # actually meaningful against this dataset

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

customers = load_csv(RAW_DIR / "customers.csv")
orders = load_csv(RAW_DIR / "orders.csv")
tickets = load_csv(RAW_DIR / "tickets.csv")

# ---------------------------------------------------------------------------
# Aggregate orders per customer
# ---------------------------------------------------------------------------

orders_by_customer = defaultdict(list)
for o in orders:
    orders_by_customer[o["customer_id"]].append(o)

tickets_by_customer = defaultdict(list)
for t in tickets:
    tickets_by_customer[t["customer_id"]].append(t)

NEG_SENTIMENTS = {"negative", "very_negative"}
RETURN_STATUSES = {"returned"}

def days_since(date_str, as_of=TODAY):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (as_of - d).days

def days_since_ts(ts_str, as_of=TODAY):
    d = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    return (as_of - d).days

rows = []
for c in customers:
    cid = c["customer_id"]
    # Only count orders that had actually occurred as of the snapshot date --
    # otherwise "future" orders in the raw data would leak into recency/spend.
    c_orders = [
        o for o in orders_by_customer.get(cid, [])
        if datetime.strptime(o["order_date"], "%Y-%m-%d") <= TODAY
    ]
    c_tickets = tickets_by_customer.get(cid, [])

    order_count = len(c_orders)
    total_spend = round(sum(float(o["amount"]) for o in c_orders), 2)
    return_count = sum(1 for o in c_orders if o["status"] in RETURN_STATUSES)
    return_rate = round(return_count / order_count, 4) if order_count else 0.0

    ticket_count = len(c_tickets)
    tickets_last_90d = sum(
        1 for t in c_tickets if days_since_ts(t["created_at"]) <= 90
    )
    negative_ticket_count = sum(1 for t in c_tickets if t["sentiment"] in NEG_SENTIMENTS)
    sentiment_load = round(negative_ticket_count / ticket_count, 4) if ticket_count else 0.0

    if order_count:
        last_order_date = max(o["order_date"] for o in c_orders)
        recency_days = days_since(last_order_date)
    else:
        recency_days = 999  # no orders at all -> treat as maximally stale

    avg_order_value = round(total_spend / order_count, 2) if order_count else 0.0

    # --- Normalize each signal to 0-1 --------------------------------------
    recency_score = min(recency_days / 180, 1.0)          # 180+ days stale = max risk
    return_score = min(return_rate / 0.5, 1.0)             # 50%+ return rate = max risk
    ticket_score = min(tickets_last_90d / 5, 1.0)           # 5+ tickets in 90d = max risk
    sentiment_score = sentiment_load                        # already 0-1

    # --- Weighted churn risk score ------------------------------------------
    # Weights are explicit and tunable -- explainable by design, not a black box.
    WEIGHTS = {"recency": 0.35, "return": 0.20, "ticket": 0.25, "sentiment": 0.20}
    churn_risk_score = round(
        WEIGHTS["recency"] * recency_score
        + WEIGHTS["return"] * return_score
        + WEIGHTS["ticket"] * ticket_score
        + WEIGHTS["sentiment"] * sentiment_score,
        3,
    )

    if churn_risk_score >= 0.6:
        risk_segment = "high"
    elif churn_risk_score >= 0.3:
        risk_segment = "medium"
    else:
        risk_segment = "low"

    # Simple CLV estimate: avg_order_value * order_count, projected forward
    # by an assumed 2-year horizon scaled by order frequency. Kept simple
    # and stated as an assumption, not disguised as a trained model.
    estimated_clv = round(avg_order_value * max(order_count, 1) * 1.6, 2)

    if order_count == 0:
        segment_label = "New/No Orders"
    elif churn_risk_score >= 0.6 and total_spend >= 3000:
        segment_label = "High-Value/At-Risk"
    elif churn_risk_score >= 0.6:
        segment_label = "Low-Value/At-Risk"
    elif total_spend >= 3000:
        segment_label = "High-Value/Stable"
    else:
        segment_label = "Standard/Stable"

    rows.append({
        "customer_id": cid,
        "order_count": order_count,
        "total_spend": total_spend,
        "avg_order_value": avg_order_value,
        "return_count": return_count,
        "return_rate": return_rate,
        "ticket_count": ticket_count,
        "tickets_last_90d": tickets_last_90d,
        "negative_ticket_count": negative_ticket_count,
        "recency_days": recency_days,
        "churn_risk_score": churn_risk_score,
        "risk_segment": risk_segment,
        "estimated_clv": estimated_clv,
        "customer_segment": segment_label,
    })

out_path = OUT_DIR / "customer_metrics.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} customer profiles to {out_path}")
from collections import Counter
seg_counts = Counter(r["risk_segment"] for r in rows)
print("Risk segment distribution:", dict(seg_counts))
label_counts = Counter(r["customer_segment"] for r in rows)
print("Customer segment distribution:", dict(label_counts))
