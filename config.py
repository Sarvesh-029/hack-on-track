"""Sach? (सच?) - Application Configuration Settings.
Reads strictly from environment / .env file with solid validation and defaults.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core LLM
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # Fact Checking & Search
    GOOGLE_FACTCHECK_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    BRAVE_API_KEY: Optional[str] = None
    SERPAPI_API_KEY: Optional[str] = None
    NEWSAPI_KEY: Optional[str] = None

    # Storage & Cache
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./sach_cache.db")
    CACHE_TTL_HOURS: int = 24

    # WhatsApp Cloud API Webhook
    WHATSAPP_TOKEN: Optional[str] = None
    WHATSAPP_VERIFY_TOKEN: str = "sach_webhook_verify_secret_token_2026"
    WHATSAPP_PHONE_NUMBER_ID: Optional[str] = None
    WHATSAPP_APP_SECRET: Optional[str] = None

    # Server & Security
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    RATE_LIMIT_PER_MINUTE: int = 60

    # File Upload limits (bytes)
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024       # 10 MB
    MAX_AUDIO_SIZE_BYTES: int = 20 * 1024 * 1024       # 20 MB
    MAX_PDF_SIZE_BYTES: int = 25 * 1024 * 1024         # 25 MB
    MAX_PDF_PAGES: int = 10

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def has_gemini(self) -> bool:
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())

    @property
    def has_search_api(self) -> bool:
        return bool(
            (self.TAVILY_API_KEY and self.TAVILY_API_KEY.strip()) or
            (self.BRAVE_API_KEY and self.BRAVE_API_KEY.strip()) or
            (self.SERPAPI_API_KEY and self.SERPAPI_API_KEY.strip()) or
            (self.GOOGLE_FACTCHECK_API_KEY and self.GOOGLE_FACTCHECK_API_KEY.strip())
        )

    @property
    def whatsapp_configured(self) -> bool:
        return bool(
            (self.WHATSAPP_TOKEN and self.WHATSAPP_TOKEN.strip()) and
            (self.WHATSAPP_PHONE_NUMBER_ID and self.WHATSAPP_PHONE_NUMBER_ID.strip())
        )


settings = Settings()
