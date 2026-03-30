import os
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

required_env_vars = [
    'POSTGRES_HOST',
    'POSTGRES_PORT',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_DB'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error("Missing required environment variables: %s", ", ".join(missing_vars))
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")


DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)


class Base(DeclarativeBase):
    pass


AsyncEngine = create_async_engine(DATABASE_URL, echo=True)

async_session_maker = async_sessionmaker(AsyncEngine, expire_on_commit=False)


async def test_db_connection() -> None:
    """Open and close one lightweight connection to validate DB connectivity."""
    try:
        async with AsyncEngine.begin() as conn:
            await conn.run_sync(lambda _: None)
        logger.info("Database connection check passed")
    except Exception as exc:
        logger.exception("Database connection check failed: %s", exc)
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
