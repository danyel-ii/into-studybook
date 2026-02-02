from __future__ import annotations

from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from .config import settings

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Session:
    with Session(engine) as session:
        yield session

# Ensure DB exists even when startup hooks are skipped in serverless runtimes.
try:
    init_db()  # ensure db exists on import
except Exception as exc:  # pragma: no cover - best effort in serverless
    print(f"DB init failed: {exc}")

