from __future__ import annotations

from app.generation import generate_syllabus, generate_tags
from app.llm import MockLLMClient
from app.schemas import TagWeight
from app.storage import append_jsonl, atomic_write_json, ensure_project_dirs, project_dir


def _seed_project(project_id: str) -> None:
    ensure_project_dirs(project_id)
    pages_path = project_dir(project_id) / "repo" / "pages.jsonl"
    append_jsonl(
        pages_path,
        [
            {
                "id": "page1",
                "url": "https://example.com",
                "source_root": "example.com",
                "fetched_at": "2024-01-01T00:00:00Z",
                "title": "Example",
                "author": "",
                "published_at": "",
                "text": "Ethereum basics and ledger concepts.",
                "language": "",
                "links_out": [],
                "checksum": "abc",
                "status": "ok",
                "error": "",
            }
        ],
    )


def test_generate_tags() -> None:
    project_id = "test-tags"
    _seed_project(project_id)
    llm = MockLLMClient()
    tags = generate_tags(project_id, llm)
    assert len(tags.tags) == 12


def test_generate_syllabus() -> None:
    project_id = "test-syllabus"
    _seed_project(project_id)
    llm = MockLLMClient()
    tags = generate_tags(project_id, llm)
    atomic_write_json(project_dir(project_id) / "tags.json", tags.model_dump())
    weights = [TagWeight(name="Tag 1", weight=1.0)]
    syllabus = generate_syllabus(project_id, weights, llm)
    assert len(syllabus.lectures) == 30
