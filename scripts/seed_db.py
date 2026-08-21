import csv
import os
from pathlib import Path

import psycopg

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://cia_user:cia_password@localhost:5432/customer_intelligence",
)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def seed_customers(cur):
    rows = load_csv(RAW_DIR / "customers.csv")
    for r in rows:
        cur.execute(
            """
            INSERT INTO customers (customer_id, name, email, tier, signup_date, country)
            VALUES (%(customer_id)s, %(name)s, %(email)s, %(tier)s, %(signup_date)s, %(country)s)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            r,
        )
    print(f"Seeded {len(rows)} customers")


def seed_products(cur):
    rows = load_csv(RAW_DIR / "products.csv")
    for r in rows:
        cur.execute(
            """
            INSERT INTO products (product_id, product_name, category, unit_price, unit_cost)
            VALUES (%(product_id)s, %(product_name)s, %(category)s, %(unit_price)s, %(unit_cost)s)
            ON CONFLICT (product_id) DO NOTHING
            """,
            r,
        )
    print(f"Seeded {len(rows)} products")


def seed_orders(cur):
    rows = load_csv(RAW_DIR / "orders.csv")
    for r in rows:
        cur.execute(
            """
            INSERT INTO orders (order_id, customer_id, product_id, order_date, quantity, amount, status)
            VALUES (%(order_id)s, %(customer_id)s, %(product_id)s, %(order_date)s, %(quantity)s, %(amount)s, %(status)s)
            ON CONFLICT (order_id) DO NOTHING
            """,
            r,
        )
    print(f"Seeded {len(rows)} orders")


def seed_tickets(cur):
    rows = load_csv(RAW_DIR / "tickets.csv")
    for r in rows:
        cur.execute(
            """
            INSERT INTO tickets (ticket_id, customer_id, ticket_type, message, sentiment, status, created_at)
            VALUES (%(ticket_id)s, %(customer_id)s, %(ticket_type)s, %(message)s, %(sentiment)s, %(status)s, %(created_at)s)
            ON CONFLICT (ticket_id) DO NOTHING
            """,
            r,
        )
    print(f"Seeded {len(rows)} tickets")


def seed_customer_metrics(cur):
    path = PROCESSED_DIR / "customer_metrics.csv"
    if not path.exists():
        print("customer_metrics.csv not found -- run scripts/compute_customer_metrics.py first")
        return
    rows = load_csv(path)
    for r in rows:
        cur.execute(
            """
            INSERT INTO customer_metrics (
                customer_id, order_count, total_spend, avg_order_value,
                return_count, return_rate, ticket_count, tickets_last_90d,
                negative_ticket_count, recency_days, churn_risk_score,
                risk_segment, estimated_clv, customer_segment
            )
            VALUES (
                %(customer_id)s, %(order_count)s, %(total_spend)s, %(avg_order_value)s,
                %(return_count)s, %(return_rate)s, %(ticket_count)s, %(tickets_last_90d)s,
                %(negative_ticket_count)s, %(recency_days)s, %(churn_risk_score)s,
                %(risk_segment)s, %(estimated_clv)s, %(customer_segment)s
            )
            ON CONFLICT (customer_id) DO UPDATE SET
                order_count = EXCLUDED.order_count,
                total_spend = EXCLUDED.total_spend,
                avg_order_value = EXCLUDED.avg_order_value,
                return_count = EXCLUDED.return_count,
                return_rate = EXCLUDED.return_rate,
                ticket_count = EXCLUDED.ticket_count,
                tickets_last_90d = EXCLUDED.tickets_last_90d,
                negative_ticket_count = EXCLUDED.negative_ticket_count,
                recency_days = EXCLUDED.recency_days,
                churn_risk_score = EXCLUDED.churn_risk_score,
                risk_segment = EXCLUDED.risk_segment,
                estimated_clv = EXCLUDED.estimated_clv,
                customer_segment = EXCLUDED.customer_segment,
                computed_at = now()
            """,
            r,
        )
    print(f"Seeded {len(rows)} customer_metrics rows")


def main():
    print(f"Connecting to {DB_URL}")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            seed_customers(cur)
            seed_products(cur)
            seed_orders(cur)
            seed_tickets(cur)
            seed_customer_metrics(cur)
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()