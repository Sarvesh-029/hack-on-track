"""Extractors for text, URLs, PDFs, images, and audio files:
1. PyMuPDF for digital PDFs (with page cap)
2. Trafilatura for ad-free web article parsing
3. Gemini multimodal for screenshots / circulars OCR
4. Gemini audio understanding for voice notes transcription + language detection
"""
import io
import re
from typing import Tuple, Optional
import pymupdf  # PyMuPDF
import trafilatura
from backend.app.core.config import settings
from backend.app.core.security import is_ssrf_safe_url

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 10) -> str:
    """Extracts text from PDF bytes using PyMuPDF up to max_pages."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages_to_read = min(len(doc), max_pages)
    text_parts = []
    for i in range(pages_to_read):
        page = doc[i]
        text = page.get_text()
        if text.strip():
            text_parts.append(f"--- Page {i+1} ---\n{text.strip()}")
    doc.close()
    return "\n\n".join(text_parts)


def extract_text_from_url(url: str) -> str:
    """Fetches and extracts clean text from a web article using Trafilatura."""
    safe, msg = is_ssrf_safe_url(url)
    if not safe:
        raise ValueError(f"SSRF Protection: {msg}")

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch content from URL: {url}")

    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False
    )
    if not extracted or not extracted.strip():
        raise ValueError(f"Could not extract article text from: {url}")

    return extracted.strip()


async def extract_text_from_image_gemini(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Extracts text / OCR from image bytes using Gemini multimodal vision."""
    if not settings.has_gemini:
        raise ValueError("GEMINI_API_KEY is not configured for image OCR.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    prompt = (
        "Extract all legible text from this image or screenshot accurately. "
        "If it is a circular, headline, WhatsApp forward, social media post, or document, transcribe the entire message word-for-word. "
        "Do not summarize or add commentary. Return ONLY the transcribed text."
    )

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt
    ]

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=contents
    )
    return response.text.strip() if response.text else ""


async def extract_text_from_audio_gemini(audio_bytes: bytes, mime_type: str = "audio/mp3") -> Tuple[str, str]:
    """Transcribes audio notes using Gemini audio understanding and detects language.
    Returns: (transcribed_text, detected_language)
    """
    if not settings.has_gemini:
        raise ValueError("GEMINI_API_KEY is not configured for audio transcription.")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    prompt = (
        "Transcribe this voice note or audio clip verbatim. "
        "Also identify the primary spoken language (e.g. Hindi, English, Marathi, Tamil, Bengali, etc.). "
        "Format your answer EXACTLY as:\n"
        "LANGUAGE: <LanguageName>\n"
        "TRANSCRIPT:\n<Verbatim transcript>"
    )

    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        prompt
    ]

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=contents
    )

    full_output = response.text.strip() if response.text else ""
    detected_lang = "English"
    transcript = full_output

    lang_match = re.search(r"LANGUAGE:\s*([^\n]+)", full_output, re.IGNORECASE)
    if lang_match:
        detected_lang = lang_match.group(1).strip()

    trans_match = re.search(r"TRANSCRIPT:\s*(.*)", full_output, re.IGNORECASE | re.DOTALL)
    if trans_match:
        transcript = trans_match.group(1).strip()

    return transcript, detected_lang


def detect_input_language(text: str) -> str:
    """Detects Indian language script or falls back to English/Hindi based on unicode characters."""
    # Devanagari (Hindi, Marathi)
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    # Bengali
    bengali_count = len(re.findall(r'[\u0980-\u09FF]', text))
    # Tamil
    tamil_count = len(re.findall(r'[\u0B80-\u0BFF]', text))
    # Telugu
    telugu_count = len(re.findall(r'[\u0C00-\u0C7F]', text))
    # Kannada
    kannada_count = len(re.findall(r'[\u0C80-\u0CFF]', text))
    # Malayalam
    malayalam_count = len(re.findall(r'[\u0D00-\u0D7F]', text))
    # Gujarati
    gujarati_count = len(re.findall(r'[\u0A80-\u0AFF]', text))
    # Gurmukhi / Punjabi
    punjabi_count = len(re.findall(r'[\u0A00-\u0A7F]', text))
    # Arabic / Urdu
    urdu_count = len(re.findall(r'[\u0600-\u06FF]', text))

    counts = [
        (devanagari_count, "Hindi"),
        (bengali_count, "Bengali"),
        (tamil_count, "Tamil"),
        (telugu_count, "Telugu"),
        (kannada_count, "Kannada"),
        (malayalam_count, "Malayalam"),
        (gujarati_count, "Gujarati"),
        (punjabi_count, "Punjabi"),
        (urdu_count, "Urdu"),
    ]

    max_script, lang = max(counts, key=lambda x: x[0])
    if max_script > 10:
        return lang

    return "English"
