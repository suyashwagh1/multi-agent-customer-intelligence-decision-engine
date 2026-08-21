import json

import psycopg
from langchain_core.tools import tool

from app.db.session import DB_URL


@tool
def log_escalation(customer_id: str, reason: str) -> str:
    """Log an escalation ticket for human agent follow-up.

    Use this whenever a conversation should be handed off to a human --
    the customer explicitly asks for a manager, reports a repeated
    unresolved issue, or the situation falls outside automated handling.
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            details = json.dumps({"action": "escalation", "reason": reason})
            cur.execute(
                """
                INSERT INTO tickets (ticket_id, customer_id, ticket_type, message,
                                      sentiment, status, details, created_at)
                VALUES (
                    'TICK-ESC-' || substr(md5(random()::text), 1, 8),
                    %s, 'escalation', %s, 'negative', 'open', %s, now()
                )
                """,
                (customer_id, reason, details),
            )
            conn.commit()

    return (
        f"This conversation has been escalated to a human support agent. "
        f"Reason logged: {reason}. Someone will follow up shortly."
    )