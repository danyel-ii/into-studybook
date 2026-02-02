from __future__ import annotations

from typing import Any

from .config import settings
from .schemas import TagWeight
from .storage import append_jsonl, project_dir, read_jsonl


def estimate_tokens_from_text(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 0.75))


def split_text_into_chunks(text: str, target_tokens: int = 1000, max_tokens: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        tokens = estimate_tokens_from_text(paragraph)
        if current_tokens + tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_tokens = tokens
        else:
            current.append(paragraph)
            current_tokens += tokens
            if current_tokens >= target_tokens:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_repository_chunks(project_id: str) -> dict[str, Any]:
    pages_path = project_dir(project_id) / "repo" / "pages.jsonl"
    chunks_path = project_dir(project_id) / "repo" / "chunks.jsonl"
    if chunks_path.exists():
        chunks_path.unlink()

    pages = read_jsonl(pages_path)
    chunk_count = 0
    for page in pages:
        if page.get("status") != "ok":
            continue
        text = page.get("text", "")
        if not text:
            continue
        for idx, chunk in enumerate(split_text_into_chunks(text)):
            record = {
                "chunk_id": f"{page['id']}-{idx}",
                "page_id": page["id"],
                "url": page.get("url", ""),
                "chunk_index": idx,
                "text": chunk,
                "token_estimate": estimate_tokens_from_text(chunk),
                "tags": [],
                "embedding_id": None,
            }
            append_jsonl(chunks_path, [record])
            chunk_count += 1

    return {"chunks": chunk_count}


def keyword_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    score = 0.0
    text_lower = text.lower()
    for term in terms:
        if not term:
            continue
        score += text_lower.count(term.lower())
    return score


def retrieve_relevant_chunks(
    project_id: str, tag_weights: list[TagWeight], limit: int = 6
) -> list[dict]:
    if settings.embeddings_enabled:
        raise RuntimeError("Embeddings are enabled but not configured in this MVP build.")

    chunks_path = project_dir(project_id) / "repo" / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)

    terms: list[str] = []
    weights: dict[str, float] = {}
    for tag_weight in tag_weights:
        if tag_weight.weight <= 0:
            continue
        term = tag_weight.name.strip().lower()
        if term:
            terms.append(term)
            weights[term] = tag_weight.weight

    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        score = 0.0
        for term in terms:
            score += keyword_score(text, [term]) * weights.get(term, 1.0)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def get_repository_stats(project_id: str) -> dict[str, int]:
    pages = read_jsonl(project_dir(project_id) / "repo" / "pages.jsonl")
    chunks = read_jsonl(project_dir(project_id) / "repo" / "chunks.jsonl")
    words = sum(len(page.get("text", "").split()) for page in pages)
    return {"pages": len(pages), "chunks": len(chunks), "words": words}
