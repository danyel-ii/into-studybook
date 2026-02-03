from __future__ import annotations

from contextlib import contextmanager

from sqlmodel import Session, SQLModel, create_engine

from .config import settings

_engine = None


def _build_engine():
    return create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )


def ensure_engine() -> None:
    global _engine
    if _engine is not None:
        return
    try:
        _engine = _build_engine()
        SQLModel.metadata.create_all(_engine)
    except Exception:
        # Fallback to /tmp if initial path is not writable in serverless.
        settings.projects_dir = Path("/tmp/selfstudy/projects")
        settings.db_path = settings.projects_dir / "metadata.db"
        settings.projects_dir.mkdir(parents=True, exist_ok=True)
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = _build_engine()
        SQLModel.metadata.create_all(_engine)


def init_db() -> None:
    ensure_engine()


@contextmanager
def get_session() -> Session:
    ensure_engine()
    with Session(_engine) as session:
        yield session
