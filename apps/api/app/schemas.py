from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Source(BaseModel):
    url: HttpUrl
    mode: Literal["single", "index", "sitemap"] = "single"
    max_pages: int = 25
    include_patterns: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None


class SourcesPayload(BaseModel):
    sources: list[Source]


class Tag(BaseModel):
    name: str
    description: str
    seed_keywords: Optional[list[str]] = None
    example_questions: Optional[list[str]] = None


class TagsPayload(BaseModel):
    tags: list[Tag]

    @model_validator(mode="after")
    def ensure_12(self) -> "TagsPayload":
        if len(self.tags) != 12:
            raise ValueError("Exactly 12 tags are required.")
        return self


class TagWeight(BaseModel):
    name: str
    weight: float = Field(ge=0.0)


class RecommendedSource(BaseModel):
    url: str
    rationale: str


class SyllabusLecture(BaseModel):
    lecture_number: int = Field(ge=1)
    title: str
    summary: str
    learning_objectives: list[str]
    key_terms: list[str]
    recommended_sources: list[RecommendedSource]
    estimated_reading_time_minutes: int
    focus_tags: list[TagWeight]


class SyllabusPayload(BaseModel):
    lectures: list[SyllabusLecture]

class SyllabusGenerateRequest(BaseModel):
    tag_weights: list[TagWeight]
    audience: Optional[str] = "general"
    difficulty: Optional[str] = "intermediate"
    lecture_count: Optional[int] = None


class TagsGenerateRequest(BaseModel):
    sample_size: int = 20


class EssayGenerateRequest(BaseModel):
    topic: str
    tag_weights: list[TagWeight]
    length: Optional[str] = "medium"


class LectureGenerateRequest(BaseModel):
    tag_weights: list[TagWeight]
    lecture_chunked_enabled: Optional[bool] = None
    lecture_chunk_parts: Optional[int] = None
    lecture_chunk_overhead_words: Optional[int] = None



class RepoStats(BaseModel):
    pages: int
    chunks: int
    words: int


class ScrapeSummary(BaseModel):
    ok: int
    skipped: int
    failed: int
    words: int
    updated_at: datetime
