from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..config import get_optional_database_url
from .session import create_engine_from_settings


def get_database_health() -> dict[str, str]:
    database_url = get_optional_database_url()
    if not database_url:
        return {"status": "not_configured"}

    try:
        engine = create_engine_from_settings()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except SQLAlchemyError:
        return {"status": "unavailable"}

    return {"status": "ok"}
