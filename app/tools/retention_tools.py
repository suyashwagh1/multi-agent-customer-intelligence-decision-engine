import json

import psycopg
from langchain_core.tools import tool

from app.db.session import DB_URL


@tool
def get_customer_profile(customer_id: str) -> str:
    """Look up a customer's risk profile and value metrics.

    Use this BEFORE offering any retention discount, to understand the
    customer's lifetime value, churn risk, and segment. Never offer a
    discount without checking this first.
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT total_spend, churn_risk_score, risk_segment,
                       customer_segment, return_rate, tickets_last_90d,
                       estimated_clv
                FROM customer_metrics
                WHERE customer_id = %s
                """,
                (customer_id,),
            )
            row = cur.fetchone()

    if row is None:
        return f"No profile found for customer {customer_id}."

    total_spend, churn_risk, risk_segment, segment, return_rate, tickets_90d, clv = row
    return (
        f"Customer {customer_id}: total spend ${total_spend}, "
        f"churn risk {churn_risk} ({risk_segment}), segment: {segment}, "
        f"return rate {return_rate}, {tickets_90d} support tickets in last 90 days, "
        f"estimated CLV ${clv}."
    )


@tool
def apply_retention_discount(customer_id: str, discount_pct: float, justification: str) -> str:
    """Apply a retention discount to a customer's account.

    Use this only after checking get_customer_profile. Per policy, max
    discount is 20%. Size the discount based on the customer's risk
    profile: high-value + high-risk customers can receive up to 20%,
    but lower-value or lower-risk customers should generally receive
    5-15%, not the maximum by default.
    """
    if discount_pct > 20:
        return (
            f"Cannot apply {discount_pct}% -- exceeds the 20% policy maximum. "
            "Escalate if a larger discount seems warranted."
        )

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            details = json.dumps({
                "action": "retention_discount",
                "discount_pct": discount_pct,
                "justification": justification,
            })
            cur.execute(
                """
                INSERT INTO tickets (ticket_id, customer_id, ticket_type, message,
                                      sentiment, status, details, created_at)
                VALUES (
                    'TICK-DISC-' || substr(md5(random()::text), 1, 8),
                    %s, 'retention_risk', %s, 'negative', 'resolved', %s, now()
                )
                """,
                (customer_id, justification, details),
            )
            conn.commit()

    return f"Applied {discount_pct}% retention discount to {customer_id}. Justification: {justification}"