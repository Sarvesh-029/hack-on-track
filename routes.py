"""Core API routes:
1. POST /api/check (multipart or json) -> returns job_id
2. GET /api/check/{job_id}/stream -> SSE live streaming of 8-step pipeline
3. GET /api/history -> List saved checks
4. GET /api/history/{id} -> Single saved check
5. DELETE /api/history/{id} -> Delete saved check
6. POST /api/re-explain -> Re-render explanation in another language
7. GET /health -> Service health status
"""
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, delete

from backend.app.core.config import settings
from backend.app.core.security import redact_pii, is_ssrf_safe_url
from backend.app.core.database import AsyncSessionLocal, HistoryRecord
from backend.app.models.schemas import (
    VerdictEnum,
    ClaimVerificationResult,
    HistoryItem,
    HistoryListResponse,
    HealthResponse
)
from backend.app.extractors.content_extractor import (
    extract_text_from_pdf,
    extract_text_from_url,
    extract_text_from_image_gemini,
    extract_text_from_audio_gemini,
    detect_input_language
)
from backend.app.services.pipeline_orchestrator import (
    run_pipeline_orchestrator,
    generate_simple_explanation_and_whatsapp_reply
)

logger = logging.getLogger("sach.routes")
router = APIRouter(prefix="/api", tags=["API"])

# In-memory job staging for streaming
JOB_STAGING = {}


class CheckTextInput(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    target_language: Optional[str] = None


class ReExplainInput(BaseModel):
    claims: List[ClaimVerificationResult]
    target_language: str


@router.post("/check")
async def create_check_job(
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    target_language: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Initiates a fact-check job. Supports text, URL, PDF, image, or audio file."""
    job_id = str(uuid.uuid4())
    raw_text = ""
    detected_type = "text"
    extracted_lang = target_language or "English"

    if file:
        file_bytes = await file.read()
        filename = (file.filename or "").lower()
        content_type = (file.content_type or "").lower()

        # Validation on file sizes
        if "pdf" in content_type or filename.endswith(".pdf"):
            if len(file_bytes) > settings.MAX_PDF_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="PDF exceeds 25 MB limit.")
            raw_text = extract_text_from_pdf(file_bytes, max_pages=settings.MAX_PDF_PAGES)
            detected_type = "pdf"
            extracted_lang = target_language or detect_input_language(raw_text)

        elif any(img in content_type for img in ["image", "png", "jpg", "jpeg", "webp"]):
            if len(file_bytes) > settings.MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit.")
            raw_text = await extract_text_from_image_gemini(file_bytes, mime_type=content_type or "image/jpeg")
            detected_type = "image"
            extracted_lang = target_language or detect_input_language(raw_text)

        elif any(aud in content_type for aud in ["audio", "ogg", "opus", "mp3", "wav", "m4a"]) or any(filename.endswith(ext) for ext in [".ogg", ".opus", ".mp3", ".wav", ".m4a"]):
            if len(file_bytes) > settings.MAX_AUDIO_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="Audio exceeds 20 MB limit.")
            raw_text, auto_lang = await extract_text_from_audio_gemini(file_bytes, mime_type=content_type or "audio/mp3")
            detected_type = "audio"
            extracted_lang = target_language or auto_lang

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {content_type}")

    elif url:
        safe, msg = is_ssrf_safe_url(url)
        if not safe:
            raise HTTPException(status_code=400, detail=f"SSRF Protection: {msg}")
        raw_text = extract_text_from_url(url)
        detected_type = "url"
        extracted_lang = target_language or detect_input_language(raw_text)

    elif text:
        raw_text = text.strip()
        detected_type = "text"
        extracted_lang = target_language or detect_input_language(raw_text)

    else:
        raise HTTPException(status_code=400, detail="No input provided. Provide text, url, or file.")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the provided input.")

    # Redact PII
    sanitized_text = redact_pii(raw_text)

    # Store in job staging
    JOB_STAGING[job_id] = {
        "job_id": job_id,
        "raw_text": sanitized_text,
        "detected_type": detected_type,
        "detected_language": extracted_lang,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    return {
        "job_id": job_id,
        "detected_type": detected_type,
        "detected_language": extracted_lang,
        "preview": sanitized_text[:120]
    }


@router.get("/check/{job_id}/stream")
async def stream_check_progress(job_id: str):
    """Streams live SSE events for the 8-step pipeline."""
    job_info = JOB_STAGING.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found or expired.")

    return StreamingResponse(
        run_pipeline_orchestrator(
            job_id=job_id,
            raw_text=job_info["raw_text"],
            detected_type=job_info["detected_type"],
            detected_language=job_info["detected_language"]
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/re-explain")
async def re_explain_claims(payload: ReExplainInput):
    """Re-renders explanation and WhatsApp reply in another language without re-running retrieval."""
    try:
        res = await generate_simple_explanation_and_whatsapp_reply(
            payload.claims,
            payload.target_language
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=HistoryListResponse)
async def list_history(
    verdict: Optional[str] = Query(None),
    limit: int = Query(20, le=100)
):
    """Lists recent verification history."""
    async with AsyncSessionLocal() as session:
        stmt = select(HistoryRecord).order_by(HistoryRecord.created_at.desc()).limit(limit)
        if verdict:
            stmt = stmt.where(HistoryRecord.overall_verdict == verdict)

        res = await session.execute(stmt)
        rows = res.scalars().all()

        items = []
        for r in rows:
            try:
                c_data = json.loads(r.claims_data_json)
            except Exception:
                c_data = []

            items.append(HistoryItem(
                id=r.id,
                created_at=r.created_at.isoformat() if r.created_at else "",
                detected_type=r.detected_type,
                detected_language=r.detected_language,
                overall_verdict=VerdictEnum(r.overall_verdict),
                claims_count=r.claims_count,
                summary_preview=r.redacted_input[:140],
                whatsapp_reply_text=r.whatsapp_reply_text,
                claims_data=c_data
            ))

        return HistoryListResponse(items=items, total=len(items))


@router.get("/history/{record_id}")
async def get_history_detail(record_id: str):
    """Retrieves full details for a history record."""
    async with AsyncSessionLocal() as session:
        stmt = select(HistoryRecord).where(HistoryRecord.id == record_id)
        res = await session.execute(stmt)
        row = res.scalars().first()
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")

        return {
            "id": row.id,
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "detected_type": row.detected_type,
            "detected_language": row.detected_language,
            "overall_verdict": row.overall_verdict,
            "claims_count": row.claims_count,
            "redacted_input": row.redacted_input,
            "simple_explanation": row.simple_explanation,
            "whatsapp_reply_text": row.whatsapp_reply_text,
            "claims": json.loads(row.claims_data_json)
        }


@router.delete("/history/{record_id}")
async def delete_history_detail(record_id: str):
    """Deletes a history record."""
    async with AsyncSessionLocal() as session:
        stmt = delete(HistoryRecord).where(HistoryRecord.id == record_id)
        await session.execute(stmt)
        await session.commit()
        return {"status": "deleted", "id": record_id}
