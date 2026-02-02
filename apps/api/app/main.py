from __future__ import annotations

import asyncio
import json
import shutil
import zipfile
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlmodel import Session, delete, select

from .config import settings
from .db import engine, init_db
from .generation import (
    _coerce_syllabus_payload,
    apply_weighted_focus,
    improve_syllabus_titles,
    generate_essay,
    generate_lectures,
    generate_syllabus,
    generate_tags,
)
from .jobs import create_job, get_job, update_job
from .llm import get_llm_client
from .models import JobRead, JobStatus, JobType, Project, ProjectCreate, ProjectRead
from .repository import build_repository_chunks, get_repository_stats
from .schemas import (
    EssayGenerateRequest,
    LectureGenerateRequest,
    RepoStats,
    SyllabusGenerateRequest,
    SyllabusPayload,
    TagsGenerateRequest,
    TagsPayload,
    SourcesPayload,
    TagWeight,
)
from .scraper import scrape_project
from .storage import (
    atomic_write_json,
    ensure_project_dirs,
    project_dir,
    read_json,
    read_jsonl,
    safe_filename,
)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _get_project(project_id: str) -> Project:
    with Session(engine) as session:
        project = session.exec(select(Project).where(Project.id == project_id)).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project


@app.post("/projects", response_model=ProjectRead)
async def create_project(payload: ProjectCreate) -> ProjectRead:
    allow_robots = (
        payload.allow_robots
        if payload.allow_robots is not None
        else settings.allow_robots_default
    )
    project = Project(name=payload.name, allow_robots=allow_robots)
    with Session(engine) as session:
        session.add(project)
        session.commit()
        session.refresh(project)
    ensure_project_dirs(project.id)
    atomic_write_json(project_dir(project.id) / "sources.json", {"sources": []})
    return ProjectRead.model_validate(project)


@app.get("/projects", response_model=list[ProjectRead])
async def list_projects() -> list[ProjectRead]:
    with Session(engine) as session:
        projects = session.exec(select(Project)).all()
    return [ProjectRead.model_validate(p) for p in projects]


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> dict[str, str]:
    project = _get_project(project_id)
    with Session(engine) as session:
        session.delete(project)
        session.commit()
    folder = project_dir(project_id)
    if folder.exists():
        shutil.rmtree(folder)
    return {"status": "ok"}


@app.delete("/projects")
async def delete_all_projects() -> dict[str, str]:
    with Session(engine) as session:
        session.exec(delete(Project))
        session.commit()
    if settings.projects_dir.exists():
        for path in settings.projects_dir.iterdir():
            if path.name == "metadata.db":
                continue
            if path.is_dir():
                shutil.rmtree(path)
    return {"status": "ok"}


@app.get("/projects/{project_id}/sources")
async def get_sources(project_id: str) -> dict[str, Any]:
    _get_project(project_id)
    return read_json(project_dir(project_id) / "sources.json", default={"sources": []})


@app.post("/projects/{project_id}/sources")
async def update_sources(project_id: str, payload: SourcesPayload) -> dict[str, Any]:
    _get_project(project_id)
    data = payload.model_dump(mode="json")
    atomic_write_json(project_dir(project_id) / "sources.json", data)
    return {"status": "ok", "sources": data["sources"]}


async def _run_scrape_job(job_id: str, project_id: str) -> None:
    update_job(job_id, status=JobStatus.running, message="starting scrape")

    def progress(current: int, total: int, message: str) -> None:
        progress_value = (current / total) if total else 0
        update_job(job_id, progress=progress_value, current_step=current, total_steps=total, message=message)

    try:
        project = _get_project(project_id)
        sources_json = read_json(project_dir(project_id) / "sources.json", default={"sources": []})
        sources = SourcesPayload.model_validate(sources_json).sources
        await scrape_project(project_id, sources, project.allow_robots, progress_cb=progress)
        update_job(job_id, status=JobStatus.succeeded, progress=1.0, message="scrape complete")
    except Exception as exc:
        update_job(job_id, status=JobStatus.failed, message=str(exc))


@app.post("/projects/{project_id}/scrape", response_model=JobRead)
async def start_scrape(project_id: str) -> JobRead:
    _get_project(project_id)
    job = create_job(project_id, JobType.scrape)
    asyncio.create_task(_run_scrape_job(job.id, project_id))
    return JobRead.model_validate(job)


async def _run_repo_build(job_id: str, project_id: str) -> None:
    update_job(job_id, status=JobStatus.running, message="building repo")
    try:
        result = build_repository_chunks(project_id)
        update_job(job_id, status=JobStatus.succeeded, progress=1.0, message=json.dumps(result))
    except Exception as exc:
        update_job(job_id, status=JobStatus.failed, message=str(exc))


@app.post("/projects/{project_id}/repo/build", response_model=JobRead)
async def build_repo(project_id: str) -> JobRead:
    _get_project(project_id)
    job = create_job(project_id, JobType.repo_build)
    asyncio.create_task(_run_repo_build(job.id, project_id))
    return JobRead.model_validate(job)


@app.get("/projects/{project_id}/scrape/summary")
async def scrape_summary(project_id: str) -> dict[str, Any]:
    _get_project(project_id)
    summary = read_json(project_dir(project_id) / "repo" / "scrape_summary.json")
    if not summary:
        return {"ok": 0, "skipped": 0, "failed": 0, "words": 0, "discovered": 0, "updated_at": ""}
    return summary


@app.get("/projects/{project_id}/scrape/errors")
async def scrape_errors(project_id: str, limit: int = 10) -> dict[str, Any]:
    _get_project(project_id)
    pages = read_jsonl(project_dir(project_id) / "repo" / "pages.jsonl")
    errors: list[dict[str, str]] = []
    for page in reversed(pages):
        if page.get("status") == "error":
            errors.append({"url": page.get("url", ""), "error": page.get("error", "")})
            if len(errors) >= limit:
                break
    return {"errors": errors}


@app.get("/projects/{project_id}/repo/stats", response_model=RepoStats)
async def get_repo_stats(project_id: str) -> RepoStats:
    _get_project(project_id)
    return RepoStats.model_validate(get_repository_stats(project_id))


@app.post("/projects/{project_id}/tags/generate", response_model=TagsPayload)
async def generate_tags_endpoint(project_id: str, payload: TagsGenerateRequest) -> TagsPayload:
    _get_project(project_id)
    llm = get_llm_client()
    try:
        return generate_tags(project_id, llm)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/projects/{project_id}/tags", response_model=TagsPayload)
async def get_tags(project_id: str) -> TagsPayload:
    _get_project(project_id)
    data = read_json(project_dir(project_id) / "tags.json")
    if not data:
        raise HTTPException(status_code=404, detail="Tags not found")
    return TagsPayload.model_validate(data)


@app.put("/projects/{project_id}/tags", response_model=TagsPayload)
async def update_tags(project_id: str, payload: TagsPayload) -> TagsPayload:
    _get_project(project_id)
    atomic_write_json(project_dir(project_id) / "tags.json", payload.model_dump())
    return payload


@app.post("/projects/{project_id}/syllabus/generate", response_model=SyllabusPayload)
async def generate_syllabus_endpoint(project_id: str, payload: SyllabusGenerateRequest) -> SyllabusPayload:
    _get_project(project_id)
    if not (project_dir(project_id) / "tags.json").exists():
        raise HTTPException(status_code=400, detail="Generate tags first")
    llm = get_llm_client()
    try:
        return generate_syllabus(project_id, payload.tag_weights, llm, payload.lecture_count)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _run_syllabus_job(
    job_id: str,
    project_id: str,
    tag_weights: list[TagWeight],
    lecture_count: int | None,
) -> None:
    update_job(job_id, status=JobStatus.running, message="generating syllabus")
    try:
        llm = get_llm_client()
        await asyncio.to_thread(generate_syllabus, project_id, tag_weights, llm, lecture_count)
        update_job(job_id, status=JobStatus.succeeded, progress=1.0, message="syllabus ready")
    except Exception as exc:
        update_job(job_id, status=JobStatus.failed, message=str(exc))


@app.post("/projects/{project_id}/syllabus/generate-job", response_model=JobRead)
async def generate_syllabus_job(project_id: str, payload: SyllabusGenerateRequest) -> JobRead:
    _get_project(project_id)
    if not (project_dir(project_id) / "tags.json").exists():
        raise HTTPException(status_code=400, detail="Generate tags first")
    job = create_job(project_id, JobType.syllabus_generate)
    asyncio.create_task(_run_syllabus_job(job.id, project_id, payload.tag_weights, payload.lecture_count))
    return JobRead.model_validate(job)


@app.get("/projects/{project_id}/syllabus/draft", response_model=SyllabusPayload)
async def get_syllabus_draft(project_id: str) -> SyllabusPayload:
    _get_project(project_id)
    data = read_json(project_dir(project_id) / "syllabus_draft.json")
    if not data:
        raise HTTPException(status_code=404, detail="Syllabus draft not found")
    lecture_count = None
    if isinstance(data, dict) and isinstance(data.get("lectures"), list):
        lecture_count = len(data["lectures"])
    payload = _coerce_syllabus_payload(data, lecture_count or settings.syllabus_lecture_count)
    tags = read_json(project_dir(project_id) / "tags.json") or {}
    tag_weights = []
    if isinstance(tags, dict):
        for t in tags.get("tags", []):
            name = t.get("name")
            weight = t.get("weight")
            if name:
                tag_weights.append(TagWeight(name=name, weight=float(weight) if weight is not None else 1.0))
    payload = apply_weighted_focus(payload, tag_weights)
    tag_names = [tw.name for tw in tag_weights]
    payload = improve_syllabus_titles(payload, tag_names)
    atomic_write_json(project_dir(project_id) / "syllabus_draft.json", payload.model_dump())
    return payload


@app.post("/projects/{project_id}/syllabus/approve")
async def approve_syllabus(project_id: str) -> dict[str, str]:
    _get_project(project_id)
    draft = read_json(project_dir(project_id) / "syllabus_draft.json")
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    atomic_write_json(project_dir(project_id) / "syllabus_approved.json", draft)
    return {"status": "ok"}


async def _run_lectures_job(
    job_id: str,
    project_id: str,
    tag_weights: list[dict],
    lecture_chunked_enabled: bool | None,
    lecture_chunk_parts: int | None,
    lecture_chunk_overhead_words: int | None,
) -> None:
    update_job(job_id, status=JobStatus.running, progress=0.0, message="generating lectures")
    progress_state = {"current": 0, "total": 0, "message": "generating lectures"}

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(10)
            total = progress_state["total"] or 0
            current = progress_state["current"]
            progress = (current / total) if total else 0.0
            update_job(
                job_id,
                status=JobStatus.running,
                progress=progress,
                current_step=current,
                total_steps=total,
                message=progress_state["message"],
            )

    hb_task = asyncio.create_task(heartbeat())
    original_chunked = settings.lecture_chunked_enabled
    original_parts = settings.lecture_chunk_parts
    original_overhead = settings.lecture_chunk_overhead_words
    if lecture_chunked_enabled is not None:
        settings.lecture_chunked_enabled = lecture_chunked_enabled
    if lecture_chunk_parts is not None:
        settings.lecture_chunk_parts = lecture_chunk_parts
    if lecture_chunk_overhead_words is not None:
        settings.lecture_chunk_overhead_words = lecture_chunk_overhead_words
    try:
        llm = get_llm_client()
        weights = [TagWeight.model_validate(item) for item in tag_weights]
        def progress_callback(current: int, total: int, title: str) -> None:
            current_step = current
            if title.startswith("Generating lecture"):
                current_step = max(current - 1, 0)
            progress_state["current"] = current_step
            progress_state["total"] = total
            progress_state["message"] = title
            progress_value = min(max(current_step / total, 0.0), 0.99) if total else 0.0
            update_job(
                job_id,
                progress=progress_value,
                current_step=current_step,
                total_steps=total,
                message=title,
            )

        outputs = await asyncio.to_thread(generate_lectures, project_id, weights, llm, progress_callback)
        update_job(job_id, status=JobStatus.succeeded, progress=1.0, message=f"{len(outputs)} lectures")
    except Exception as exc:
        update_job(job_id, status=JobStatus.failed, message=str(exc))
    finally:
        hb_task.cancel()
        settings.lecture_chunked_enabled = original_chunked
        settings.lecture_chunk_parts = original_parts
        settings.lecture_chunk_overhead_words = original_overhead


@app.post("/projects/{project_id}/lectures/generate", response_model=JobRead)
async def generate_lectures_endpoint(project_id: str, payload: LectureGenerateRequest) -> JobRead:
    _get_project(project_id)
    if not (project_dir(project_id) / "syllabus_approved.json").exists():
        raise HTTPException(status_code=400, detail="Approve syllabus first")
    job = create_job(project_id, JobType.lectures_generate)
    asyncio.create_task(
        _run_lectures_job(
            job.id,
            project_id,
            [tw.model_dump() for tw in payload.tag_weights],
            payload.lecture_chunked_enabled,
            payload.lecture_chunk_parts,
            payload.lecture_chunk_overhead_words,
        )
    )
    return JobRead.model_validate(job)


@app.get("/projects/{project_id}/lectures")
async def list_lectures(project_id: str) -> dict[str, Any]:
    _get_project(project_id)
    folder = project_dir(project_id) / "outputs" / "lectures"
    files = sorted([p.name for p in folder.glob("*.md")])
    return {"files": files}


@app.get("/projects/{project_id}/lectures/{lecture_number}")
async def get_lecture(project_id: str, lecture_number: int) -> dict[str, str]:
    _get_project(project_id)
    folder = project_dir(project_id) / "outputs" / "lectures"
    matches = sorted(folder.glob(f"{lecture_number:02d}-*.md"))
    if not matches:
        raise HTTPException(status_code=404, detail="Lecture not found")
    content = matches[0].read_text(encoding="utf-8")
    return {"content": content}




@app.delete("/projects/{project_id}/lectures")
async def purge_lectures(project_id: str) -> dict[str, int]:
    _get_project(project_id)
    folder = project_dir(project_id) / "outputs" / "lectures"
    deleted = 0
    if folder.exists():
        for path in folder.glob("*.md"):
            path.unlink(missing_ok=True)
            deleted += 1
    return {"deleted": deleted}


@app.get("/projects/{project_id}/downloads/lectures.zip")
async def download_lectures(project_id: str) -> FileResponse:
    _get_project(project_id)
    folder = project_dir(project_id) / "outputs" / "lectures"
    zip_path = project_dir(project_id) / "outputs" / "lectures.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in folder.glob("*.md"):
            zf.write(file, arcname=file.name)
    return FileResponse(zip_path, filename="lectures.zip")


@app.delete("/projects/{project_id}/repo")
async def delete_repo(project_id: str) -> dict[str, str]:
    _get_project(project_id)
    repo_path = project_dir(project_id) / "repo"
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.mkdir(parents=True, exist_ok=True)
    return {"status": "ok"}


@app.delete("/projects/{project_id}/cache")
async def delete_cache(project_id: str) -> dict[str, str]:
    _get_project(project_id)
    cache_path = project_dir(project_id) / "cache"
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    return {"status": "ok"}


@app.post("/projects/{project_id}/essays/generate")
async def generate_essay_endpoint(project_id: str, payload: EssayGenerateRequest) -> dict[str, str]:
    _get_project(project_id)
    llm = get_llm_client()
    path = generate_essay(project_id, payload.topic, payload.tag_weights, llm)
    return {"path": str(path)}


@app.get("/projects/{project_id}/essays")
async def list_essays(project_id: str) -> dict[str, Any]:
    _get_project(project_id)
    folder = project_dir(project_id) / "outputs" / "essays"
    files = sorted([p.name for p in folder.glob("*.md")])
    return {"files": files}


@app.get("/projects/{project_id}/essays/{essay_name}")
async def get_essay(project_id: str, essay_name: str) -> dict[str, str]:
    _get_project(project_id)
    try:
        safe_name = safe_filename(essay_name)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid essay name")
    path = project_dir(project_id) / "outputs" / "essays" / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Essay not found")
    return {"content": path.read_text(encoding="utf-8")}


@app.get("/jobs/{job_id}", response_model=JobRead)
async def get_job_status(job_id: str) -> JobRead:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobRead.model_validate(job)