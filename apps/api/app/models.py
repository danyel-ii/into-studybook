from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlmodel import Field, SQLModel


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobType(str, Enum):
    scrape = "scrape"
    repo_build = "repo_build"
    tags_generate = "tags_generate"
    syllabus_generate = "syllabus_generate"
    lectures_generate = "lectures_generate"
    essay_generate = "essay_generate"


class Project(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    name: str
    allow_robots: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Job(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    project_id: str = Field(index=True)
    job_type: JobType
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    total_steps: int = 0
    current_step: int = 0
    message: Optional[str] = None
    result_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectCreate(BaseModel):
    name: str
    allow_robots: Optional[bool] = True


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    allow_robots: bool
    created_at: datetime


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    job_type: JobType
    status: JobStatus
    progress: float
    total_steps: int
    current_step: int
    message: Optional[str] = None
    result_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime
