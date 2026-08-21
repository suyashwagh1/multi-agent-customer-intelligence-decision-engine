import json
import uuid
from datetime import datetime

import psycopg
from langchain_core.tools import tool

from app.db.session import DB_URL


@tool
def create_return_request(order_id: str, customer_id: str, reason: str) -> str:
    """Create a return request for a customer's order.

    Use this when a customer wants to return an item, says it arrived
    damaged, or asks for a refund on a specific order. Requires the
    order_id, the customer_id who placed it, and a brief reason for
    the return.
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # Confirm the order actually exists and belongs to this customer
            cur.execute(
                "SELECT status FROM orders WHERE order_id = %s AND customer_id = %s",
                (order_id, customer_id),
            )
            row = cur.fetchone()
            if row is None:
                return (
                    f"Could not find order {order_id} for customer {customer_id}. "
                    "Please verify both IDs."
                )

            order_status = row[0]
            ticket_id = f"TICK-RET-{uuid.uuid4().hex[:8].upper()}"
            details = json.dumps({
                "action": "return_request",
                "order_id": order_id,
                "reason": reason,
                "order_status_at_request": order_status,
            })

            cur.execute(
                """
                INSERT INTO tickets (ticket_id, customer_id, order_id, ticket_type,
                                      message, sentiment, status, details, created_at)
                VALUES (%s, %s, %s, 'return_request', %s, 'neutral', 'open', %s, %s)
                """,
                (ticket_id, customer_id, order_id, reason, details, datetime.now()),
            )
            conn.commit()

    return (
        f"Return request {ticket_id} created for order {order_id}. "
        f"Reason: {reason}. A confirmation and return label will follow."
    )