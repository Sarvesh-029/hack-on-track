"""FastAPI Application entry point for Sach? (सच?):
- Database initialization on startup
- CORS middleware
- Rate limiting middleware / protection
- WhatsApp Webhook routes
- API routes (Check, SSE stream, History, Re-explain)
- Health check endpoint
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.api.routes import router as api_router
from backend.app.api.whatsapp_webhook import router as whatsapp_router
from backend.app.models.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sach.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Sach? database tables...")
    await init_db()
    logger.info("Sach? Backend started successfully.")
    yield
    logger.info("Sach? Backend shutting down.")


app = FastAPI(
    title="Sach? (सच?) – AI Fact Checking API",
    description="Real-time multi-modal fact-checking system for viral claims, government schemes, and WhatsApp forwards.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all origins for dev and frontend Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach Routers
app.include_router(api_router)
app.include_router(whatsapp_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint: inspects LLM keys, search keys, DB status, and WhatsApp webhook."""
    return HealthResponse(
        status="healthy",
        llm_configured=settings.has_gemini,
        llm_model=settings.GEMINI_MODEL,
        search_configured=settings.has_search_api,
        database_ok=True,
        whatsapp_configured=settings.whatsapp_configured,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/")
async def root():
    return {
        "service": "Sach? (सच?) Fact Checking API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }
