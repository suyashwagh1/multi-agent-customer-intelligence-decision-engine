# Escalation Rules

## When to Escalate to a Human Agent
A conversation should be escalated to human support when any of the
following are true:
- The customer explicitly requests a manager or human agent.
- The customer reports the same unresolved issue across multiple prior
  contacts (2 or more previous tickets on the same order or issue).
- The sentiment of the conversation is classified as very_negative.
- The requested action falls outside any agent's defined tool
  capabilities (e.g. a legal dispute, a chargeback already filed with the
  customer's bank, or a request involving a data privacy concern).
- The AI agent's confidence in the correct intent classification falls
  below the acceptable threshold, rather than guessing.

## Escalation Is Not a Failure State
Escalating is the correct outcome, not a fallback of last resort, whenever
a case matches the criteria above. An agent that resolves an ambiguous or
high-stakes case incorrectly causes more harm than one that hands off
cleanly with context.

## What Gets Logged on Escalation
When a case is escalated, the system should log: the customer's original
message, the classified intent and confidence score, any actions already
taken (e.g. a partial refund already applied), and a one-line reason for
the escalation, so the human agent does not have to re-discover context.

## Retention-Specific Escalation
Retention-risk conversations that include a specific dollar threshold (for
example, a customer citing total lifetime spend as a reason to leave) or
that come from a customer already at the maximum allowed retention
discount should be escalated rather than having a discount reapplied.
