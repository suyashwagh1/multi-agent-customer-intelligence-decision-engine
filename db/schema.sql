CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS customers (
    customer_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'standard',
    signup_date     DATE NOT NULL,
    country         TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id      TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT,
    unit_price      NUMERIC(10, 2) NOT NULL,
    unit_cost       NUMERIC(10, 2)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    product_id      TEXT NOT NULL REFERENCES products(product_id),
    order_date      DATE NOT NULL,
    quantity        INTEGER NOT NULL DEFAULT 1,
    amount          NUMERIC(10, 2) NOT NULL,
    status          TEXT NOT NULL DEFAULT 'processing'
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    order_id        TEXT REFERENCES orders(order_id),
    ticket_type     TEXT NOT NULL,
    message         TEXT NOT NULL,
    sentiment       TEXT NOT NULL DEFAULT 'neutral',
    status          TEXT NOT NULL DEFAULT 'open',
    details         JSONB,
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS customer_metrics (
    customer_id             TEXT PRIMARY KEY REFERENCES customers(customer_id),
    order_count             INTEGER NOT NULL DEFAULT 0,
    total_spend             NUMERIC(10, 2) NOT NULL DEFAULT 0,
    avg_order_value         NUMERIC(10, 2) NOT NULL DEFAULT 0,
    return_count            INTEGER NOT NULL DEFAULT 0,
    return_rate             NUMERIC(5, 4) NOT NULL DEFAULT 0,
    ticket_count            INTEGER NOT NULL DEFAULT 0,
    tickets_last_90d        INTEGER NOT NULL DEFAULT 0,
    negative_ticket_count   INTEGER NOT NULL DEFAULT 0,
    recency_days            INTEGER NOT NULL DEFAULT 0,
    churn_risk_score        NUMERIC(5, 3) NOT NULL DEFAULT 0,
    risk_segment            TEXT NOT NULL DEFAULT 'low',
    estimated_clv           NUMERIC(10, 2) NOT NULL DEFAULT 0,
    customer_segment        TEXT NOT NULL DEFAULT 'Standard/Stable',
    computed_at             TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id        SERIAL PRIMARY KEY,
    doc_filename    TEXT NOT NULL,
    doc_title       TEXT NOT NULL,
    section_title   TEXT,
    content         TEXT NOT NULL,
    embedding       vector(384)
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_customer_id ON tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at);
CREATE INDEX IF NOT EXISTS idx_customer_metrics_risk ON customer_metrics(risk_segment);
