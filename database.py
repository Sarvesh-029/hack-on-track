"""Database models & session management using SQLAlchemy 2.0 and aiosqlite/asyncpg.
Provides:
1. Search query cache with 24-hour TTL
2. Verification history store with redacted inputs
"""
import json
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Float, Integer

from backend.app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


class SearchCache(Base):
    __tablename__ = "search_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    results_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    def is_expired(self, ttl_hours: int = 24) -> bool:
        now = datetime.now(timezone.utc)
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created) > timedelta(hours=ttl_hours)


class HistoryRecord(Base):
    __tablename__ = "history_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    detected_type: Mapped[str] = mapped_column(String(32))
    detected_language: Mapped[str] = mapped_column(String(32))
    redacted_input: Mapped[str] = mapped_column(Text)
    overall_verdict: Mapped[str] = mapped_column(String(32), index=True)
    claims_count: Mapped[int] = mapped_column(Integer, default=1)
    simple_explanation: Mapped[str] = mapped_column(Text)
    whatsapp_reply_text: Mapped[str] = mapped_column(Text)
    claims_data_json: Mapped[str] = mapped_column(Text)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
