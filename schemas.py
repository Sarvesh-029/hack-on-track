"""Pydantic schemas strictly enforcing the fact-checking pipeline requirements:
- Verdicts: VERIFIED, FALSE, OUTDATED, PARTLY SUPPORTED, CANNOT BE CONFIRMED
- Anti-hallucination citation guard
- Multi-claim breakdown
- Multilingual explanation + shareable WhatsApp reply
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class VerdictEnum(str, Enum):
    VERIFIED = "VERIFIED"
    FALSE = "FALSE"
    OUTDATED = "OUTDATED"
    PARTLY_SUPPORTED = "PARTLY SUPPORTED"
    CANNOT_BE_CONFIRMED = "CANNOT BE CONFIRMED"


class StanceEnum(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceCard(BaseModel):
    title: str = Field(..., description="Page or article title")
    publisher: str = Field(..., description="Name of the news or gov publisher")
    url: str = Field(..., description="Real source URL")
    date: Optional[str] = Field(None, description="Publication or update date, ISO or year")
    stance: StanceEnum = Field(..., description="supports, contradicts, or neutral")
    quote_under_25_words: str = Field(
        ...,
        description="Direct quote or concise snippet under 25 words from the page",
        max_length=250
    )


class AtomicClaim(BaseModel):
    id: str = Field(..., description="e.g. C1, C2")
    original_wording: str = Field(..., description="Original claim phrasing from the forward")
    normalized_query: str = Field(..., description="Normalized English search query")


class ClaimVerificationResult(BaseModel):
    claim_id: str
    original_claim: str
    normalized_query: str
    verdict: VerdictEnum
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    what_is_wrong: str = Field(..., description="Explanation of what is incorrect or unverified")
    why: str = Field(..., description="Reasoning and factual breakdown")
    what_is_correct: str = Field(..., description="The verified real fact or context")
    evidence: List[EvidenceCard] = Field(default_factory=list)
    last_verified_date: Optional[str] = None


class VerificationJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    detected_type: str
    detected_language: str
    claims_count: int = 0
    overall_verdict: Optional[VerdictEnum] = None
    simple_explanation: Optional[str] = None
    whatsapp_reply_text: Optional[str] = None
    claims: List[ClaimVerificationResult] = Field(default_factory=list)
    error: Optional[str] = None


class HistoryItem(BaseModel):
    id: str
    created_at: str
    detected_type: str
    detected_language: str
    overall_verdict: VerdictEnum
    claims_count: int
    summary_preview: str
    whatsapp_reply_text: str
    claims_data: List[ClaimVerificationResult]


class HistoryListResponse(BaseModel):
    items: List[HistoryItem]
    total: int


class WhatsAppWebhookVerification(BaseModel):
    hub_mode: Optional[str] = None
    hub_verify_token: Optional[str] = None
    hub_challenge: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    llm_model: str
    search_configured: bool
    database_ok: bool
    whatsapp_configured: bool
    timestamp: str
