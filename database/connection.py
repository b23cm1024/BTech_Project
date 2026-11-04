"""
database/connection.py
======================
SQLAlchemy database engine and session factory supporting both PostgreSQL
(production) and SQLite (local zero-dependency fallback).
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("DatabaseConnection")

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "iitj_academic.db"))
os.makedirs(os.path.dirname(db_path), exist_ok=True)
sqlite_url = f"sqlite:///{db_path}"

target_url = os.getenv("DATABASE_URL")
engine = None

# If external PostgreSQL URL is provided, test connectivity before committing
if target_url and not os.getenv("FORCE_SQLITE"):
    try:
        test_engine = create_engine(target_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with test_engine.connect() as conn:
            pass
        engine = test_engine
        logger.info(f"Connected to relational database at {target_url.split('@')[-1] if '@' in target_url else target_url}")
    except Exception as e:
        logger.warning(f"Could not connect to configured DATABASE_URL ({e}). Falling back to local SQLite: {sqlite_url}")
        engine = None

if engine is None:
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
    logger.info(f"Using local SQLite database at {db_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
