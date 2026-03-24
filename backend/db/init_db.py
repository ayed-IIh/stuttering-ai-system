"""
Database initialization script.

IMPORTANT:
This file is intended for LOCAL DEVELOPMENT ONLY.

In production environments the database schema must be created using
the SQL migration file:

    backend/db/migrations/001_initial_schema.sql

Do NOT rely on SQLAlchemy create_all() in production.
"""

from .database import AsyncEngine
from .database import Base
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def init_db():
    """Create all tables using SQLAlchemy metadata. Used only for local development."""
    try:
        logger.info("Creating database tables...")
        async with AsyncEngine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Tables created successfully.")
    except Exception as e:
        logger.info(f"❌ Failed to create tables: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
