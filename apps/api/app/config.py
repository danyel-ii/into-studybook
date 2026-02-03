from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[3]



def _default_projects_dir() -> Path:
    if (
        os.getenv("VERCEL")
        or os.getenv("VERCEL_ENV")
        or os.getenv("VERCEL_URL")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
    ):
        return Path("/tmp/selfstudy/projects")
    return ROOT_DIR / "projects"


class Settings(BaseSettings):
    app_name: str = "EthEd Notebook Builder API"
    projects_dir: Path = _default_projects_dir()
    db_path: Optional[Path] = None

    allow_robots_default: bool = True
    requests_per_host: float = 1.0
    max_concurrency: int = 4
    http_timeout: float = 20.0
    max_retries: int = 2
    user_agent: str = "EthEdNotebookBot/0.1 (+local)"

    environment: str = "development"

    llm_provider: str = "mock"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.0

    embeddings_enabled: bool = False

    dspy_enabled: bool = True
    dspy_max_revisions: int = 2
    allow_mock_fallback: bool = False
    syllabus_context_chunks: int = 16
    syllabus_lecture_count: int = 30
    lecture_target_words: int = 7500
    lecture_word_tolerance: float = 0.15
    lecture_min_words: int = 6000
    lecture_max_words: int = 9000
    lecture_max_list_items: int = 3
    lecture_max_expansion_rounds: int = 6
    lecture_chunked_enabled: bool = True
    lecture_chunk_parts: int = 3
    lecture_chunk_overhead_words: int = 900
    test_mode: bool = False
    test_lecture_min_words: int = 200
    test_lecture_max_words: int = 400
    test_lecture_target_words: int = 300
    test_lecture_word_tolerance: float = 0.5
    test_lecture_chunk_parts: int = 1
    test_lecture_chunk_overhead_words: int = 80

    class Config:
        env_file = (
            ".env",
            str(ROOT_DIR / ".env"),
            str(ROOT_DIR / "apps" / "api" / ".env"),
        )
        env_prefix = ""




def _clean_openai_key(value: str) -> str:
    cleaned = value.strip().strip(""").strip("'")
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned.split(" ", 1)[1].strip()
    return cleaned


settings = Settings()
if settings.openai_api_key:
    settings.openai_api_key = _clean_openai_key(settings.openai_api_key)

if settings.db_path is None:
    settings.db_path = settings.projects_dir / "metadata.db"

try:
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
except OSError:
    settings.projects_dir = Path("/tmp/selfstudy/projects")
    settings.db_path = settings.projects_dir / "metadata.db"
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

if settings.environment.lower().strip() in {"production", "prod"}:
    settings.allow_mock_fallback = False
