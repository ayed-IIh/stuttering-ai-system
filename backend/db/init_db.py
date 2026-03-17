"""
Database initialization script.

IMPORTANT:
This file is intended for LOCAL DEVELOPMENT ONLY.

In production environments the database schema must be created using
the SQL migration file:

    backend/db/migrations/001_initial_schema.sql

Do NOT rely on SQLAlchemy create_all() in production.
"""

from backend.db.database import AsyncEngine
from backend.db.models import Base


async def init_db():
    """Create all tables using SQLAlchemy metadata. Used only for local development."""
    print("Creating database tables...")
    async with AsyncEngine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
