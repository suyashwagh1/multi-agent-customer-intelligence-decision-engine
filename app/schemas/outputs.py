from typing import Literal

from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    """Structured intent classification for an incoming customer message."""

    intent: Literal["order_inquiry", "retention_risk", "policy_question", "escalation"] = Field(
        description="The single best-matching category for this message."
    )
    confidence: float = Field(
        description="Confidence in this classification, from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )