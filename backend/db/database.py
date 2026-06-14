import os
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REQUIRED_DB_ENV_VARS = [
    'POSTGRES_HOST',
    'POSTGRES_PORT',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_DB',
]

# The database is OPTIONAL. It is used only to log prediction rows; inference
# works without it. Boot must not fail when Postgres is unconfigured (e.g. a
# stateless, inference-only deployment), so a missing var disables the DB
# instead of raising at import time.
_missing_vars = [var for var in REQUIRED_DB_ENV_VARS if not os.getenv(var)]

# SQL echo (verbose query logging) is off by default; opt in via DB_ECHO=true.
_DB_ECHO = os.getenv("DB_ECHO", "false").strip().lower() in {"1", "true", "yes", "on"}


class Base(DeclarativeBase):
    pass


if _missing_vars:
    logger.warning(
        "Database disabled — missing env vars: %s. Predictions will not be "
        "persisted (inference is unaffected).",
        ", ".join(_missing_vars),
    )
    DATABASE_URL = None
    AsyncEngine = None
    async_session_maker = None
    DB_ENABLED = False
else:
    try:
        _port = int(os.getenv("POSTGRES_PORT", ""))
    except ValueError as exc:
        raise EnvironmentError(
            f"POSTGRES_PORT must be an integer, got {os.getenv('POSTGRES_PORT')!r}"
        ) from exc
    # URL.create escapes every component, so credentials/host with reserved
    # characters (@ : / # etc.) can't corrupt the DSN.
    DATABASE_URL = URL.create(
        "postgresql+asyncpg",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=_port,
        database=os.getenv("POSTGRES_DB"),
    )
    AsyncEngine = create_async_engine(DATABASE_URL, echo=_DB_ECHO)
    async_session_maker = async_sessionmaker(AsyncEngine, expire_on_commit=False)
    DB_ENABLED = True


async def test_db_connection() -> None:
    """Open and close one lightweight connection to validate DB connectivity."""
    if not DB_ENABLED:
        logger.info("Database disabled; skipping connection check")
        return
    try:
        async with AsyncEngine.begin() as conn:
            await conn.run_sync(lambda _: None)
        logger.info("Database connection check passed")
    except Exception as exc:
        logger.exception("Database connection check failed: %s", exc)
        raise


async def get_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    """Yield a DB session, or ``None`` when the database is disabled.

    Callers must tolerate a ``None`` session and skip persistence in that case.
    """
    if not DB_ENABLED:
        yield None
        return
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
