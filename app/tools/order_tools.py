import psycopg
from langchain_core.tools import tool

from app.db.session import DB_URL


@tool
def check_order_status(order_id: str) -> str:
    """Look up the current status of a customer's order by order ID.

    Use this whenever a customer asks about the status, location, or
    delivery of a specific order. Requires the order_id (format: ORD-XXXXXX).
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.order_id, o.status, o.order_date, o.amount,
                       p.product_name, o.customer_id
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                WHERE o.order_id = %s
                """,
                (order_id,),
            )
            row = cur.fetchone()

    if row is None:
        return f"No order found with ID {order_id}. Please verify the order ID and try again."

    order_id, status, order_date, amount, product_name, customer_id = row
    return (
        f"Order {order_id} ({product_name}, ${amount}, placed {order_date}) "
        f"is currently '{status}'."
    )