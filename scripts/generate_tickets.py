"""
Generate realistic support tickets with controlled linguistic variation.

Instead of ~20 canned messages repeated hundreds of times, this builds
each message from templates with slot-filling (order IDs, day counts,
product names pulled from real data) so the Router agent has to parse
intent from language, not memorize fixed strings.

Deterministic: fixed seed, same output every run.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
N_TICKETS = 2500

# ---------------------------------------------------------------------------
# Load real customers/orders so tickets reference actual IDs
# ---------------------------------------------------------------------------

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

customers = load_csv(DATA_DIR / "customers.csv")
orders = load_csv(DATA_DIR / "orders.csv")
customer_ids = [c["customer_id"] for c in customers]
order_ids = [o["order_id"] for o in orders]

# ---------------------------------------------------------------------------
# Template banks per intent — each is a function so slots differ every call
# ---------------------------------------------------------------------------

def rand_days():
    return random.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 14])

def rand_order():
    return random.choice(order_ids)

ORDER_INQUIRY = [
    lambda: f"My package was supposed to arrive {rand_days()} days ago and I still haven't received it.",
    lambda: "Tracking says shipped but there hasn't been an update in four days.",
    lambda: f"Can someone tell me what's happening with order {rand_order()}?",
    lambda: f"I placed this order almost {rand_days()} days ago. Is it still in transit?",
    lambda: "The tracking page hasn't changed since it left the warehouse.",
    lambda: f"Where is order {rand_order()}? It's later than the estimate I was given.",
    lambda: "I never got a shipping confirmation email, did my order actually go through?",
    lambda: "The courier app says delivered but nothing showed up at my door.",
    lambda: f"Checking in on {rand_order()} — no movement on tracking for almost a week.",
    lambda: "Is there a delay on my shipment? It's been quiet since it left the facility.",
    lambda: "My delivery window passed yesterday and there's no update.",
    lambda: f"Order {rand_order()} shows 'processing' for {rand_days()} days now, is that normal?",
]

RETURN_REQUEST = [
    lambda: f"The item I got in order {rand_order()} isn't what I ordered, I need a replacement or refund.",
    lambda: "This arrived damaged, the box was crushed and the product inside is cracked.",
    lambda: f"I'd like to send this back, it's been {rand_days()} days since delivery, is that still within the window?",
    lambda: "Wrong size arrived, can I exchange it instead of a full return?",
    lambda: f"Requesting a return on {rand_order()}, the item doesn't match the listing photos.",
    lambda: "The product stopped working after two uses, I want to return it.",
    lambda: "Can I get a prepaid return label for this order?",
    lambda: f"Item from {rand_order()} arrived defective right out of the box.",
    lambda: "I ordered the wrong color by mistake, is exchange possible at this point?",
    lambda: "This isn't what was pictured on the product page, requesting a return.",
]

RETENTION_RISK = [
    lambda: "This is the third time an order has been late, I'm thinking about cancelling my account.",
    lambda: "Honestly considering switching to a different service after this experience.",
    lambda: f"If order {rand_order()} is late again I'm done ordering from here.",
    lambda: "I've been a loyal customer for years and this keeps happening, I'm losing patience.",
    lambda: "Not sure it's worth staying a member if service keeps slipping like this.",
    lambda: "Between the delays and the return hassle, I'm reconsidering my subscription.",
    lambda: "This is becoming a pattern and I don't think I want to keep ordering here.",
    lambda: "I'm one bad experience away from cancelling, just being honest.",
    lambda: "Every order this month has had some issue, what's going on?",
    lambda: "Thinking about closing my account, can you give me a reason not to?",
]

POLICY_QUESTION = [
    lambda: "What is your refund policy for items damaged in shipping?",
    lambda: f"My order arrived {rand_days()} days ago but it's damaged, can I still get a refund?",
    lambda: "How long do I have to return an item after delivery?",
    lambda: "Do premium members get extended return windows?",
    lambda: "Is expedited shipping refundable if the delivery is late anyway?",
    lambda: "What's the process for a refund on a final-sale item that arrived broken?",
    lambda: "Can I return an item I opened but didn't use?",
    lambda: "What counts as eligible for a late-delivery refund?",
    lambda: "Are digital or downloadable products ever returnable?",
    lambda: "How long does a refund actually take to process once approved?",
]

ESCALATION = [
    lambda: "I've contacted support three times about this and nobody has resolved it, I need a manager.",
    lambda: f"This is regarding order {rand_order()}, I've been going back and forth for a week with no resolution.",
    lambda: "I want to speak to someone who can actually make a decision on this.",
    lambda: "This has gone on too long, please escalate this to a supervisor.",
    lambda: "I'm not satisfied with the previous response, please escalate.",
    lambda: "Nothing offered so far has actually solved my problem, I need this escalated.",
    lambda: "I've been patient but this needs to go to someone higher up now.",
]

INTENT_BANKS = {
    "order_inquiry": ORDER_INQUIRY,
    "return_request": RETURN_REQUEST,
    "retention_risk": RETENTION_RISK,
    "policy_question": POLICY_QUESTION,
    "escalation": ESCALATION,
}

# Roughly match the original distribution: order_inquiry 30%, policy 24%,
# retention 18%, return 18%, escalation 10%
INTENT_WEIGHTS = {
    "order_inquiry": 0.30,
    "policy_question": 0.24,
    "retention_risk": 0.18,
    "return_request": 0.18,
    "escalation": 0.10,
}

SENTIMENT_BY_INTENT = {
    "order_inquiry": ["neutral", "neutral", "negative"],
    "return_request": ["neutral", "negative"],
    "retention_risk": ["negative", "very_negative"],
    "policy_question": ["neutral"],
    "escalation": ["negative", "very_negative"],
}

STATUS_CHOICES = ["open", "resolved"]

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def weighted_intent():
    intents = list(INTENT_WEIGHTS.keys())
    weights = list(INTENT_WEIGHTS.values())
    return random.choices(intents, weights=weights, k=1)[0]

def random_timestamp():
    start = datetime(2025, 1, 1)
    delta_days = random.randint(0, 260)
    dt = start + timedelta(days=delta_days, hours=random.randint(0, 23))
    return dt.strftime("%Y-%m-%dT%H:00:00")

rows = []
for i in range(1, N_TICKETS + 1):
    intent = weighted_intent()
    message = random.choice(INTENT_BANKS[intent])()
    rows.append({
        "ticket_id": f"TICK-{i:06d}",
        "customer_id": random.choice(customer_ids),
        "ticket_type": intent,
        "message": message,
        "sentiment": random.choice(SENTIMENT_BY_INTENT[intent]),
        "status": random.choice(STATUS_CHOICES),
        "created_at": random_timestamp(),
    })

out_path = DATA_DIR / "tickets.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

unique_messages = len(set(r["message"] for r in rows))
print(f"Wrote {len(rows)} tickets to {out_path}")
print(f"Unique messages: {unique_messages} (template pool: "
      f"{sum(len(v) for v in INTENT_BANKS.values())} base templates x random slot-filling)")
