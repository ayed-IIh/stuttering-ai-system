import os
import logging
from typing import AsyncGenerator, Optional
from urllib.parse import quote
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
    DATABASE_URL: Optional[str] = None
    AsyncEngine = None
    async_session_maker = None
    DB_ENABLED = False
else:
    # URL-escape the user/password/db so credentials containing reserved
    # characters (@ : / # etc.) don't corrupt the DSN.
    _user = quote(os.getenv("POSTGRES_USER", ""), safe="")
    _password = quote(os.getenv("POSTGRES_PASSWORD", ""), safe="")
    _database = quote(os.getenv("POSTGRES_DB", ""), safe="")
    DATABASE_URL = (
        f"postgresql+asyncpg://{_user}:{_password}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{_database}"
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
