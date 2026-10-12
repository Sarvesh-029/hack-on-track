"""Meta WhatsApp Cloud API Webhook Handler:
1. GET /webhook/whatsapp: verify-token handshake and challenge response
2. POST /webhook/whatsapp: processes text, image, audio, document messages
3. Downloads incoming media with access token
4. Executes Sach? fact checking pipeline
5. Replies with compact WhatsApp message (emoji + verdict + 2-line reason + top source link)
"""
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional
import httpx
from fastapi import APIRouter, Request, Response, HTTPException, status
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.security import redact_pii
from backend.app.extractors.content_extractor import (
    extract_text_from_pdf,
    extract_text_from_image_gemini,
    extract_text_from_audio_gemini,
    detect_input_language
)
from backend.app.services.pipeline_orchestrator import (
    extract_atomic_claims_gemini,
    verify_claim_with_llm,
    generate_simple_explanation_and_whatsapp_reply
)
from backend.app.retrieval.evidence_retriever import retrieve_evidence_for_claim

logger = logging.getLogger("sach.whatsapp")
router = APIRouter(prefix="/webhook/whatsapp", tags=["WhatsApp Webhook"])


def verify_whatsapp_signature(body: bytes, signature_header: Optional[str]) -> bool:
    """Validates X-Hub-Signature-256 if WHATSAPP_APP_SECRET is set."""
    if not settings.WHATSAPP_APP_SECRET or not signature_header:
        return True  # If no secret configured, skip strict signature check

    if not signature_header.startswith("sha256="):
        return False

    expected_sig = hmac.new(
        settings.WHATSAPP_APP_SECRET.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256
    ).hexdigest()

    received_sig = signature_header[7:]
    return hmac.compare_digest(expected_sig, received_sig)


@router.get("")
async def whatsapp_webhook_verification(request: Request):
    """WhatsApp Cloud API webhook verification handshake."""
    params = request.query_params
    hub_mode = params.get("hub.mode")
    hub_token = params.get("hub.verify_token")
    hub_challenge = params.get("hub.challenge")

    if hub_mode == "subscribe" and hub_token == settings.WHATSAPP_VERIFY_TOKEN:
        logger.info("WhatsApp webhook verified successfully.")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.warning("WhatsApp webhook verification token mismatch.")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Verification token mismatch")


async def download_whatsapp_media(media_id: str) -> bytes:
    """Downloads media from WhatsApp Graph API using the media ID and WHATSAPP_TOKEN."""
    if not settings.WHATSAPP_TOKEN:
        raise ValueError("WHATSAPP_TOKEN not configured.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Query media url
        meta_url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN}"}
        resp = await client.get(meta_url, headers=headers)
        if resp.status_code != 200:
            raise ValueError(f"Failed to fetch media metadata from WhatsApp: {resp.text}")

        download_url = resp.json().get("url")
        if not download_url:
            raise ValueError("No download URL returned by WhatsApp.")

        # Step 2: Download media bytes
        media_resp = await client.get(download_url, headers=headers)
        if media_resp.status_code != 200:
            raise ValueError(f"Failed to download media content: {media_resp.status_code}")

        return media_resp.content


async def send_whatsapp_reply(recipient_phone: str, message_text: str):
    """Sends a WhatsApp text message reply via Meta Cloud API."""
    if not settings.whatsapp_configured:
        logger.warning(f"WhatsApp reply skipped: Not configured. Target: {recipient_phone}")
        return

    send_url = f"https://graph.facebook.com/v19.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "type": "text",
        "text": {"preview_url": True, "body": message_text}
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(send_url, headers=headers, json=payload)
            if resp.status_code not in (200, 201):
                logger.error(f"WhatsApp reply API error: {resp.text}")
            else:
                logger.info(f"WhatsApp reply delivered to {recipient_phone}")
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")


@router.post("")
async def receive_whatsapp_message(request: Request):
    """Handles incoming message payloads from WhatsApp users."""
    body_bytes = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    if not verify_whatsapp_signature(body_bytes, sig):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    try:
        payload = await request.json()
    except Exception:
        return {"status": "ignored"}

    # Extract message entries
    entries = payload.get("entry", [])
    for entry in entries:
        for change in entry.get("changes", []):
            val = change.get("value", {})
            messages = val.get("messages", [])
            for msg in messages:
                sender = msg.get("from")
                msg_type = msg.get("type")
                raw_text = ""
                detected_lang = "English"

                try:
                    if msg_type == "text":
                        raw_text = msg.get("text", {}).get("body", "")
                        detected_lang = detect_input_language(raw_text)

                    elif msg_type == "image":
                        image_id = msg.get("image", {}).get("id")
                        mime_type = msg.get("image", {}).get("mime_type", "image/jpeg")
                        if image_id:
                            img_bytes = await download_whatsapp_media(image_id)
                            raw_text = await extract_text_from_image_gemini(img_bytes, mime_type)
                            detected_lang = detect_input_language(raw_text)

                    elif msg_type in ("audio", "voice"):
                        audio_id = msg.get("audio", {}).get("id") or msg.get("voice", {}).get("id")
                        mime_type = msg.get("audio", {}).get("mime_type", "audio/ogg")
                        if audio_id:
                            audio_bytes = await download_whatsapp_media(audio_id)
                            raw_text, detected_lang = await extract_text_from_audio_gemini(audio_bytes, mime_type)

                    elif msg_type == "document":
                        doc_id = msg.get("document", {}).get("id")
                        if doc_id:
                            doc_bytes = await download_whatsapp_media(doc_id)
                            raw_text = extract_text_from_pdf(doc_bytes)
                            detected_lang = detect_input_language(raw_text)

                    # Redact PII
                    clean_text = redact_pii(raw_text)

                    if clean_text.strip():
                        # Run pipeline
                        claims = await extract_atomic_claims_gemini(clean_text)
                        verified_claims = []
                        for c in claims:
                            srcs = await retrieve_evidence_for_claim(c.normalized_query)
                            v_res = await verify_claim_with_llm(c, srcs)
                            verified_claims.append(v_res)

                        explanation = await generate_simple_explanation_and_whatsapp_reply(
                            verified_claims,
                            detected_lang
                        )

                        # Reply to sender
                        reply_text = explanation.get("whatsapp_reply_text", "Could not verify this message.")
                        await send_whatsapp_reply(sender, reply_text)
                    else:
                        await send_whatsapp_reply(
                            sender,
                            "❓ Sach? Bot: Could not read text from this message to verify. Please send text, photo, audio or document."
                        )

                except Exception as e:
                    logger.error(f"Error handling WhatsApp message: {e}")
                    await send_whatsapp_reply(
                        sender,
                        f"⚠️ Sach? Error: Verification failed. ({str(e)[:100]})"
                    )

    return {"status": "processed"}
