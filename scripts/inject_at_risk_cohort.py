"""
Inject a deliberate High-Value/At-Risk cohort.

The random generation produced only 1 customer in the High-Value/At-Risk
segment -- too thin to demo the RetentionAgent's core reasoning case
("high LTV + high churn risk -> smaller justified discount, not the max").

This script targets a subset of genuinely high-spend customers and gives
them the signal pattern that should realistically put them at risk:
recent negative-sentiment tickets and at least one return. It does NOT
touch churn_risk_score or risk_segment directly -- those stay fully
derived by compute_customer_metrics.py. This script only edits the raw
inputs (orders.csv, tickets.csv); the metrics pipeline is re-run after.

Deterministic: fixed seed, same output every run.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
N_TARGET_CUSTOMERS = 25  # how many customers to push into the at-risk cohort

TODAY = datetime(2025, 9, 30)  # must match the snapshot date used in compute_customer_metrics.py

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

customers = load_csv(DATA_DIR / "customers.csv")
orders = load_csv(DATA_DIR / "orders.csv")
tickets = load_csv(DATA_DIR / "tickets.csv")

# ---------------------------------------------------------------------------
# 1. Identify top spenders using the SAME as-of filter the metrics pipeline
#    uses -- otherwise we select customers whose spend includes orders that
#    get excluded downstream, and the cohort never actually clears the
#    High-Value threshold.
# ---------------------------------------------------------------------------

spend_by_customer = {}
for o in orders:
    if datetime.strptime(o["order_date"], "%Y-%m-%d") <= TODAY:
        spend_by_customer[o["customer_id"]] = spend_by_customer.get(o["customer_id"], 0) + float(o["amount"])

# Sort descending by (as-of) spend, take a slice from the top -- want real
# variety in the cohort, so sample from the qualifying pool rather than
# always grabbing exactly the top N
ranked = sorted(spend_by_customer.items(), key=lambda kv: kv[1], reverse=True)
qualifying = [cid for cid, spend in ranked if spend >= 3000]  # must clear High-Value threshold on its own
top_pool = qualifying if len(qualifying) >= N_TARGET_CUSTOMERS else [cid for cid, _ in ranked[:N_TARGET_CUSTOMERS * 2]]

random.shuffle(top_pool)
at_risk_cohort = top_pool[:N_TARGET_CUSTOMERS]

print(f"Selected {len(at_risk_cohort)} high-spend customers for at-risk injection")
print("Spend range in cohort:", 
      round(min(spend_by_customer[c] for c in at_risk_cohort), 2), "-",
      round(max(spend_by_customer[c] for c in at_risk_cohort), 2))

# ---------------------------------------------------------------------------
# 2. Flip one of each cohort customer's orders to 'returned' if none exists
# ---------------------------------------------------------------------------

orders_by_customer = {}
for o in orders:
    orders_by_customer.setdefault(o["customer_id"], []).append(o)

flipped = 0
for cid in at_risk_cohort:
    c_orders = orders_by_customer.get(cid, [])
    non_cancelled_non_returned = [o for o in c_orders if o["status"] not in ("cancelled", "returned")]
    # Flip up to 2 orders to 'returned' -- need return_rate meaningfully above
    # 0 to move the return-risk component; one flip is often too weak against
    # a customer with 8-10 total orders.
    n_to_flip = min(2, len(non_cancelled_non_returned))
    for target in random.sample(non_cancelled_non_returned, n_to_flip):
        target["status"] = "returned"
        flipped += 1

print(f"Flipped {flipped} orders to 'returned' status for cohort customers")

# ---------------------------------------------------------------------------
# 3. Add recent negative-sentiment tickets for the cohort
# ---------------------------------------------------------------------------

RETENTION_TEMPLATES = [
    "I've spent a lot with you over the years but this keeps happening, I'm close to done.",
    "Given how much I order here, I expected better -- seriously considering switching.",
    "This is the second issue this month on a big order, I'm losing patience.",
    "I'm a long-time customer and this experience has been disappointing lately.",
    "Not sure I want to keep placing large orders if this is the level of service.",
]

ESCALATION_TEMPLATES = [
    "This is regarding a significant order and I still haven't gotten a resolution.",
    "I've reached out before about this exact issue with no real fix.",
    "Given the size of my order, I expected this to be handled faster.",
]

NEAR_TODAY = datetime(2025, 9, 30)

def recent_timestamp():
    delta_days = random.randint(0, 45)  # within the 90-day window used by the metrics pipeline
    dt = NEAR_TODAY - timedelta(days=delta_days, hours=random.randint(0, 23))
    return dt.strftime("%Y-%m-%dT%H:00:00")

existing_ids = [int(t["ticket_id"].split("-")[1]) for t in tickets]
next_id = max(existing_ids) + 1

new_tickets = []
for cid in at_risk_cohort:
    n_new = random.randint(5, 7)  # enough to clear tickets_last_90d's cap of 5 (max ticket_score)
    for _ in range(n_new):
        is_escalation = random.random() < 0.3
        bank = ESCALATION_TEMPLATES if is_escalation else RETENTION_TEMPLATES
        new_tickets.append({
            "ticket_id": f"TICK-{next_id:06d}",
            "customer_id": cid,
            "ticket_type": "escalation" if is_escalation else "retention_risk",
            "message": random.choice(bank),
            "sentiment": random.choices(["negative", "very_negative"], weights=[0.4, 0.6])[0],
            "status": random.choice(["open", "resolved"]),
            "created_at": recent_timestamp(),
        })
        next_id += 1

print(f"Added {len(new_tickets)} new negative-sentiment tickets for cohort customers")

all_tickets = tickets + new_tickets

# ---------------------------------------------------------------------------
# Write back
# ---------------------------------------------------------------------------

write_csv(DATA_DIR / "orders.csv", orders, fieldnames=list(orders[0].keys()))
write_csv(DATA_DIR / "tickets.csv", all_tickets, fieldnames=list(all_tickets[0].keys()))

print(f"Updated orders.csv ({len(orders)} rows) and tickets.csv ({len(all_tickets)} rows)")
print("Cohort customer_ids:", at_risk_cohort)
