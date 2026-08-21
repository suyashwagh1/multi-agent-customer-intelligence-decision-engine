import os
from pathlib import Path

import psycopg

SCHEMA_PATH = Path(__file__).parent.parent / "db" / "schema.sql"

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://cia_user:cia_password@localhost:5432/customer_intelligence",
)


def apply_schema():
    sql = SCHEMA_PATH.read_text(encoding="utf-8-sig")  # strips BOM if present
    print(f"Applying schema to: {DB_URL.split('@')[-1]}")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Schema applied successfully.")


if __name__ == "__main__":
    apply_schema()