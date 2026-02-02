from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "EthEd Notebook Builder API"
    projects_dir: Path = ROOT_DIR / "projects"
    db_path: Path = projects_dir / "metadata.db"

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


settings = Settings()
settings.projects_dir.mkdir(parents=True, exist_ok=True)
settings.db_path.parent.mkdir(parents=True, exist_ok=True)

if settings.environment.lower().strip() in {"production", "prod"}:
    settings.allow_mock_fallback = False
