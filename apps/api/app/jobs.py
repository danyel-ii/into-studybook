from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Session, select

from .db import engine
from .models import Job, JobStatus, JobType


def create_job(project_id: str, job_type: JobType, total_steps: int = 0) -> Job:
    job = Job(project_id=project_id, job_type=job_type, total_steps=total_steps)
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    return job


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    progress: Optional[float] = None,
    total_steps: Optional[int] = None,
    current_step: Optional[int] = None,
    message: Optional[str] = None,
    result_path: Optional[str] = None,
) -> Job:
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).first()
        if not job:
            raise ValueError("Job not found")
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if total_steps is not None:
            job.total_steps = total_steps
        if current_step is not None:
            job.current_step = current_step
        if message is not None:
            job.message = message
        if result_path is not None:
            job.result_path = result_path
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def get_job(job_id: str) -> Optional[Job]:
    with Session(engine) as session:
        return session.exec(select(Job).where(Job.id == job_id)).first()
