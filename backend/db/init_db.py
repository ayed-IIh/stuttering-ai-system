"""
Database initialization script.

IMPORTANT:
This file is intended for LOCAL DEVELOPMENT ONLY.

In production environments the database schema must be created using
the SQL migration file:

    backend/db/migrations/001_initial_schema.sql

Do NOT rely on SQLAlchemy create_all() in production.
"""

import logging

from backend.db.database import AsyncEngine, Base


logger = logging.getLogger(__name__)


async def init_db():
    """Create all tables using SQLAlchemy metadata. Used only for local development."""
    try:
        logger.info("Creating database tables for local development")
        async with AsyncEngine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as exc:
        logger.exception("Failed to create database tables: %s", exc)
        raise


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(init_db())
