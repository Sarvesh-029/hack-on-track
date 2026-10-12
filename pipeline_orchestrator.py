"""Verification Orchestrator using Real LLM Calls (Gemini) + Pydantic validation:
- Step 3: Claim extraction (1-5 atomic claims, normalized English query, short ID)
- Step 4: Parallel Retrieval
- Step 5: Verification with anti-hallucination guard
- Step 6: Multi-lingual explanation (Grade-6 reading level)
- Step 7: WhatsApp reply generation
- Step 8: Persistence & SSE events streaming
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, AsyncGenerator, Optional
from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.models.schemas import (
    AtomicClaim,
    ClaimVerificationResult,
    EvidenceCard,
    VerdictEnum,
    StanceEnum,
    VerificationJobResponse,
)
from backend.app.retrieval.evidence_retriever import retrieve_evidence_for_claim, RetrievedSource
from backend.app.core.database import AsyncSessionLocal, HistoryRecord

logger = logging.getLogger("sach.orchestrator")

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def get_gemini_client():
    if not settings.has_gemini:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please provide a valid GEMINI_API_KEY in the .env file. "
            "No fake data or mock responses are permitted."
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


async def extract_atomic_claims_gemini(text: str) -> List[AtomicClaim]:
    """Uses Gemini to pull out 1–5 atomic, checkable claims from the forward.
    Ignores chain-letter fluff, greetings, emojis, opinions.
    """
    client = get_gemini_client()

    system_prompt = (
        "You are an expert fact-checking analyst for viral messages and forwards. "
        "Your task is to extract 1 to 5 atomic, verifiable factual claims from the text. "
        "Ignore greetings, blessings, chain-letter threats ('forward to 10 people'), rhetorical questions, and personal opinions. "
        "For each claim, assign an ID (C1, C2..), provide the exact original wording, and write a concise, normalized English search query for fact-checking databases.\n"
        "Return ONLY a JSON array conforming to this schema:\n"
        "[\n"
        "  {\"id\": \"C1\", \"original_wording\": \"...\", \"normalized_query\": \"...\"}\n"
        "]"
    )

    prompt = f"Text to analyze:\n```\n{text}\n```"

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[system_prompt, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    raw_text = response.text.strip() if response.text else "[]"
    try:
        parsed = json.loads(raw_text)
        claims = [AtomicClaim(**item) for item in parsed]
        if not claims:
            # Fallback single claim if text itself was very concise
            claims = [AtomicClaim(
                id="C1",
                original_wording=text[:200],
                normalized_query=text[:120]
            )]
        return claims[:5]
    except Exception as e:
        logger.error(f"Failed to parse atomic claims from LLM output: {raw_text}. Error: {e}")
        # Clean regex fallback
        return [AtomicClaim(id="C1", original_wording=text[:200], normalized_query=text[:120])]


async def verify_claim_with_llm(claim: AtomicClaim, sources: List[RetrievedSource]) -> ClaimVerificationResult:
    """Verifies a claim against genuine retrieved source texts.
    Enforces Anti-Hallucination Guard: Every cited URL must strictly exist in the retrieved evidence set.
    """
    # HARD RULE: If retrieval returns nothing, verdict must be CANNOT BE CONFIRMED.
    if not sources:
        return ClaimVerificationResult(
            claim_id=claim.id,
            original_claim=claim.original_wording,
            normalized_query=claim.normalized_query,
            verdict=VerdictEnum.CANNOT_BE_CONFIRMED,
            confidence=0.5,
            what_is_wrong="Insufficient verifiable evidence found.",
            why="No authoritative reports, government notices, or established fact-checks confirmed or refuted this claim.",
            what_is_correct="The claim cannot be independently verified with the available trusted public records.",
            evidence=[],
            last_verified_date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )

    client = get_gemini_client()

    # Build genuine evidence payload for the LLM
    evidence_context = []
    valid_urls_map = {}
    for idx, s in enumerate(sources):
        valid_urls_map[s.url] = s
        snippet_text = s.page_text if s.page_text else s.snippet
        evidence_context.append(
            f"SOURCE [{idx+1}]:\n"
            f"URL: {s.url}\n"
            f"Publisher: {s.publisher}\n"
            f"Title: {s.title}\n"
            f"Date: {s.date or 'Unknown'}\n"
            f"Content: {snippet_text[:1200]}\n"
        )

    context_str = "\n".join(evidence_context)

    system_instruction = (
        "You are an objective fact-checking judge for Sach? (सच?).\n"
        "Analyze the provided CLAIM against the RETRIEVED SOURCES strictly.\n"
        "Rules:\n"
        "1. Never give a bare true/false. Always explain WHAT is wrong and WHY.\n"
        "2. Clearly pick ONE verdict from: VERIFIED, FALSE, OUTDATED, PARTLY SUPPORTED, CANNOT BE CONFIRMED.\n"
        "   - VERIFIED: multiple reliable sources agree and nothing contradicts.\n"
        "   - FALSE: reliable sources directly contradict it, or a fact-checker has debunked it.\n"
        "   - OUTDATED: was once true, but newer information supersedes it (explain dates).\n"
        "   - PARTLY SUPPORTED: some parts are right, some wrong or exaggerated (detail both).\n"
        "   - CANNOT BE CONFIRMED: not enough reliable evidence in the sources.\n"
        "3. CRITICAL ANTI-HALLUCINATION RULE: In the 'evidence' array, ONLY cite URLs that appear in the PROVIDED SOURCES. "
        "Any URL not present in the sources is strictly forbidden. Quotes must be direct and under 25 words.\n"
        "4. Return ONLY valid JSON matching this schema:\n"
        "{\n"
        "  \"verdict\": \"VERIFIED\" | \"FALSE\" | \"OUTDATED\" | \"PARTLY SUPPORTED\" | \"CANNOT BE CONFIRMED\",\n"
        "  \"confidence\": 0.95,\n"
        "  \"what_is_wrong\": \"...\",\n"
        "  \"why\": \"...\",\n"
        "  \"what_is_correct\": \"...\",\n"
        "  \"evidence\": [\n"
        "    {\n"
        "      \"title\": \"...\",\n"
        "      \"publisher\": \"...\",\n"
        "      \"url\": \"... (MUST be exact match from provided sources)\",\n"
        "      \"date\": \"...\",\n"
        "      \"stance\": \"supports\" | \"contradicts\" | \"neutral\",\n"
        "      \"quote_under_25_words\": \"...\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"CLAIM TO VERIFY:\n\"{claim.original_wording}\"\n\n"
        f"RETRIEVED SOURCES:\n{context_str}"
    )

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[system_instruction, user_prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    raw_json = response.text.strip() if response.text else "{}"
    try:
        data = json.loads(raw_json)
        verdict = VerdictEnum(data.get("verdict", VerdictEnum.CANNOT_BE_CONFIRMED.value))
        confidence = float(data.get("confidence", 0.5))
        what_is_wrong = str(data.get("what_is_wrong", "Claim verification inconclusive."))
        why = str(data.get("why", "Based on the retrieved sources."))
        what_is_correct = str(data.get("what_is_correct", "Check official records."))

        # Anti-hallucination URL verification guard: drop any URL not in valid_urls_map
        validated_evidence: List[EvidenceCard] = []
        for ev in data.get("evidence", []):
            ev_url = ev.get("url", "").strip()
            if ev_url in valid_urls_map:
                real_src = valid_urls_map[ev_url]
                try:
                    stance = StanceEnum(ev.get("stance", "neutral").lower())
                except ValueError:
                    stance = StanceEnum.NEUTRAL

                quote = ev.get("quote_under_25_words", "")
                # Ensure quote is trimmed under 25 words
                words = quote.split()
                if len(words) > 25:
                    quote = " ".join(words[:25]) + "..."

                validated_evidence.append(EvidenceCard(
                    title=ev.get("title") or real_src.title,
                    publisher=ev.get("publisher") or real_src.publisher,
                    url=ev_url,
                    date=ev.get("date") or real_src.date,
                    stance=stance,
                    quote_under_25_words=quote or real_src.snippet[:150]
                ))

        # If LLM didn't attach evidence cards despite having sources, inject top source
        if not validated_evidence and sources:
            top_s = sources[0]
            validated_evidence.append(EvidenceCard(
                title=top_s.title,
                publisher=top_s.publisher,
                url=top_s.url,
                date=top_s.date,
                stance=StanceEnum.NEUTRAL,
                quote_under_25_words=top_s.snippet[:150]
            ))

        return ClaimVerificationResult(
            claim_id=claim.id,
            original_claim=claim.original_wording,
            normalized_query=claim.normalized_query,
            verdict=verdict,
            confidence=max(0.1, min(1.0, confidence)),
            what_is_wrong=what_is_wrong,
            why=why,
            what_is_correct=what_is_correct,
            evidence=validated_evidence,
            last_verified_date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
    except Exception as e:
        logger.error(f"Error validating claim verification result: {e}. Raw: {raw_json}")
        # Return graceful Cannot Be Confirmed rather than crash
        return ClaimVerificationResult(
            claim_id=claim.id,
            original_claim=claim.original_wording,
            normalized_query=claim.normalized_query,
            verdict=VerdictEnum.CANNOT_BE_CONFIRMED,
            confidence=0.5,
            what_is_wrong="Could not verify definitively from available records.",
            why=f"Data verification error: {str(e)}",
            what_is_correct="Verify with official portal.",
            evidence=[],
            last_verified_date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )


async def generate_simple_explanation_and_whatsapp_reply(
    claims: List[ClaimVerificationResult],
    target_language: str
) -> Dict[str, str]:
    """Generates:
    1. Simple 2-3 sentence explanation at Grade-6 reading level in the target language.
    2. WhatsApp shareable reply text with emoji, verdict label, reasons, and top source link.
    """
    client = get_gemini_client()

    summary_bullets = []
    sources_list = []
    verdicts = [c.verdict for c in claims]

    # Calculate overall verdict
    if VerdictEnum.FALSE in verdicts:
        overall_verdict = VerdictEnum.FALSE
    elif VerdictEnum.OUTDATED in verdicts:
        overall_verdict = VerdictEnum.OUTDATED
    elif VerdictEnum.PARTLY_SUPPORTED in verdicts:
        overall_verdict = VerdictEnum.PARTLY_SUPPORTED
    elif all(v == VerdictEnum.VERIFIED for v in verdicts):
        overall_verdict = VerdictEnum.VERIFIED
    else:
        overall_verdict = VerdictEnum.CANNOT_BE_CONFIRMED

    for c in claims:
        summary_bullets.append(f"- Claim: {c.original_claim}\n  Verdict: {c.verdict.value}\n  Fact: {c.what_is_correct}")
        for ev in c.evidence:
            if ev.url not in sources_list:
                sources_list.append(f"{ev.publisher}: {ev.url}")

    prompt = (
        f"You are Sach? (सच?), an accessible Indian fact-checking system.\n"
        f"Overall Verdict: {overall_verdict.value}\n"
        f"Claim Findings:\n" + "\n".join(summary_bullets) + "\n"
        f"Available Sources:\n" + "\n".join(sources_list[:3]) + "\n\n"
        f"Target Language: {target_language}\n"
        "Generate two outputs in valid JSON format:\n"
        "1. simple_explanation: 2 to 3 short sentences, Grade-6 reading level, in the requested Target Language. Explains clearly what is true or false without technical jargon.\n"
        "2. whatsapp_reply_text: A concise, polite message formatted for WhatsApp to paste into a family or community group. Use appropriate emoji (❌ for False, ✅ for Verified, ⚠️ for Partly Supported/Outdated, ❓ for Cannot Confirm), verdict label, a 2-line explanation in the target language, and 1-2 source links.\n\n"
        "Output JSON schema:\n"
        "{\n"
        "  \"overall_verdict\": \"" + overall_verdict.value + "\",\n"
        "  \"simple_explanation\": \"...\",\n"
        "  \"whatsapp_reply_text\": \"...\"\n"
        "}"
    )

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        data = json.loads(response.text.strip())
        return {
            "overall_verdict": overall_verdict.value,
            "simple_explanation": data.get("simple_explanation", ""),
            "whatsapp_reply_text": data.get("whatsapp_reply_text", "")
        }
    except Exception:
        # Fallback
        emoji_map = {
            VerdictEnum.VERIFIED: "✅ VERIFIED",
            VerdictEnum.FALSE: "❌ FALSE",
            VerdictEnum.OUTDATED: "⚠️ OUTDATED",
            VerdictEnum.PARTLY_SUPPORTED: "⚠️ PARTLY SUPPORTED",
            VerdictEnum.CANNOT_BE_CONFIRMED: "❓ CANNOT BE CONFIRMED"
        }
        fallback_msg = f"{emoji_map.get(overall_verdict, '❓')}: Please be careful with this forward. "
        if sources_list:
            fallback_msg += f"Verified source: {sources_list[0]}"
        return {
            "overall_verdict": overall_verdict.value,
            "simple_explanation": f"This message is {overall_verdict.value.lower()}. Please check official sources.",
            "whatsapp_reply_text": fallback_msg
        }


async def run_pipeline_orchestrator(
    job_id: str,
    raw_text: str,
    detected_type: str,
    detected_language: str
) -> AsyncGenerator[str, None]:
    """Runs the 8-step pipeline and yields Server-Sent Events (SSE) data chunks.
    Steps streamed:
    1. Step 1 & 2: Ingest & Extract done
    2. Step 3: Claim extraction
    3. Step 4: Parallel Retrieval
    4. Step 5: Claim Verification
    5. Step 6: Simple language explanation
    6. Step 7: WhatsApp reply generation
    7. Step 8: Persist to History & Complete
    """
    def sse(event: str, data: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("progress", {
        "step": 1,
        "step_name": "Ingest & Extract",
        "message": f"Extracted text from {detected_type}. Detected language: {detected_language}."
    })

    yield sse("progress", {
        "step": 2,
        "step_name": "Redaction",
        "message": "PII redaction complete (phone numbers and emails sanitized)."
    })

    yield sse("progress", {
        "step": 3,
        "step_name": "Finding Claims",
        "message": "Extracting atomic verifiable claims with Gemini..."
    })

    claims: List[AtomicClaim] = []
    try:
        claims = await extract_atomic_claims_gemini(raw_text)
        yield sse("progress", {
            "step": 3,
            "step_name": "Claims Found",
            "message": f"Identified {len(claims)} claim(s) to verify.",
            "claims": [c.model_dump() for c in claims]
        })
    except Exception as e:
        yield sse("error", {"message": f"Failed during claim extraction: {str(e)}"})
        return

    yield sse("progress", {
        "step": 4,
        "step_name": "Searching Sources",
        "message": "Querying Fact Check APIs, government portals, and trusted news databases in parallel..."
    })

    # Parallel retrieval per claim
    claim_sources: Dict[str, List[RetrievedSource]] = {}
    for c in claims:
        srcs = await retrieve_evidence_for_claim(c.normalized_query)
        claim_sources[c.id] = srcs

    total_sources = sum(len(s) for s in claim_sources.values())
    yield sse("progress", {
        "step": 4,
        "step_name": "Sources Retrieved",
        "message": f"Retrieved {total_sources} live evidence sources across all claims."
    })

    yield sse("progress", {
        "step": 5,
        "step_name": "Verifying Claims",
        "message": "Analyzing evidence, stances, and quotes with anti-hallucination validation..."
    })

    verified_claims: List[ClaimVerificationResult] = []
    for c in claims:
        res = await verify_claim_with_llm(c, claim_sources.get(c.id, []))
        verified_claims.append(res)
        yield sse("claim_progress", {
            "claim_id": c.id,
            "verdict": res.verdict.value,
            "confidence": res.confidence
        })

    yield sse("progress", {
        "step": 6,
        "step_name": "Explaining",
        "message": f"Synthesizing simple language explanation in {detected_language}..."
    })

    explanation_data = await generate_simple_explanation_and_whatsapp_reply(
        verified_claims,
        detected_language
    )

    overall_verdict_str = explanation_data["overall_verdict"]
    simple_explanation = explanation_data["simple_explanation"]
    whatsapp_reply_text = explanation_data["whatsapp_reply_text"]

    # Persist to History Database (Step 8)
    try:
        async with AsyncSessionLocal() as session:
            history_record = HistoryRecord(
                id=job_id,
                created_at=datetime.now(timezone.utc),
                detected_type=detected_type,
                detected_language=detected_language,
                redacted_input=raw_text[:2000],
                overall_verdict=overall_verdict_str,
                claims_count=len(verified_claims),
                simple_explanation=simple_explanation,
                whatsapp_reply_text=whatsapp_reply_text,
                claims_data_json=json.dumps([c.model_dump() for c in verified_claims])
            )
            session.add(history_record)
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to persist history: {e}")

    final_payload = VerificationJobResponse(
        job_id=job_id,
        status="completed",
        created_at=datetime.now(timezone.utc).isoformat(),
        detected_type=detected_type,
        detected_language=detected_language,
        claims_count=len(verified_claims),
        overall_verdict=VerdictEnum(overall_verdict_str),
        simple_explanation=simple_explanation,
        whatsapp_reply_text=whatsapp_reply_text,
        claims=verified_claims
    )

    yield sse("complete", final_payload.model_dump())
