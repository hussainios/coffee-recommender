from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_database_url


def create_engine_from_settings() -> Engine:
    return create_engine(get_database_url(), future=True)


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    bound_engine = engine or create_engine_from_settings()
    return sessionmaker(
        bind=bound_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
