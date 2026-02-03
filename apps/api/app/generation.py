from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from .config import settings
from .dspy_pipeline import PipelineResult, get_generation_pipeline
from .llm import LLMClient, MockLLMClient, get_llm_client
from .repository import retrieve_relevant_chunks
from .runs import save_run
from .schemas import TagsPayload, SyllabusPayload, TagWeight, SyllabusLecture
from .storage import atomic_write_json, iso_now, project_dir, read_json, read_jsonl, slugify


def _sample_context(project_id: str, limit: int = 6) -> str:
    chunks = read_jsonl(project_dir(project_id) / "repo" / "chunks.jsonl")
    if not chunks:
        pages = read_jsonl(project_dir(project_id) / "repo" / "pages.jsonl")
        chunks = [{"text": p.get("text", ""), "url": p.get("url", "")} for p in pages]
    samples = chunks[:limit]
    context_parts = []
    for idx, chunk in enumerate(samples, start=1):
        text = chunk.get("text", "")[:1500]
        url = chunk.get("url", "")
        context_parts.append(f"[Source {idx}] {url}\n{text}")
    return "\n\n".join(context_parts)

def _build_syllabus_context(project_id: str, tag_weights: list[TagWeight]) -> str:
    focus = tag_weights
    if not focus:
        tags_data = read_json(project_dir(project_id) / "tags.json") or {}
        focus = [TagWeight(name=t.get("name", ""), weight=1.0) for t in tags_data.get("tags", [])]
    chunks = retrieve_relevant_chunks(project_id, focus, limit=settings.syllabus_context_chunks)
    if not chunks:
        return _sample_context(project_id, limit=6)
    context_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")[:1500]
        url = chunk.get("url", "")
        context_parts.append(f"[Source {idx}] {url}\n{text}")
    return "\n\n".join(context_parts)

def _sources_context(project_id: str) -> str:
    data = read_json(project_dir(project_id) / "sources.json", default={}) or {}
    sources = data.get("sources") or []
    if not sources:
        return ""
    lines = []
    for idx, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            url = source.get("url", "")
            mode = source.get("mode") or source.get("discovery_mode") or ""
            extra = f" ({mode})" if mode else ""
        else:
            url = str(source)
            extra = ""
        if not url:
            continue
        lines.append(f"[Source {idx}] {url}{extra}")
    if not lines:
        return ""
    return "Sources list (no scraped content yet):\n" + "\n".join(lines)




def _extract_json_block(text: str) -> str | None:
    fence = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def _clean_json(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text).strip()


def _coerce_tags_payload(data: object) -> TagsPayload:
    if isinstance(data, dict) and isinstance(data.get("tags"), list):
        tags = data["tags"]
        if tags and isinstance(tags[0], str):
            data = {
                "tags": [
                    {
                        "name": tag,
                        "description": f"{tag} concepts and materials.",
                        "seed_keywords": [],
                        "example_questions": [],
                    }
                    for tag in tags
                ]
            }
    elif isinstance(data, list) and data and isinstance(data[0], str):
        data = {
            "tags": [
                {
                    "name": tag,
                    "description": f"{tag} concepts and materials.",
                    "seed_keywords": [],
                    "example_questions": [],
                }
                for tag in data
            ]
        }
    return TagsPayload.model_validate(data)


def _ensure_title(idx: int, lecture: dict) -> str:
    title = str(lecture.get("title", "")).strip()
    normalized = title.lower()
    generic_pattern = re.compile(r"^lecture\s*\d+\s*[:\-–—]?\s*$", re.IGNORECASE)
    if not title or normalized == "lecture" or normalized == f"lecture {idx}" or generic_pattern.match(title):
        focus = ""
        if lecture.get("summary"):
            focus = str(lecture.get("summary")).strip().split(".")[0]
        if not focus and lecture.get("key_terms"):
            focus = ", ".join([str(t) for t in lecture.get("key_terms", [])][:3])
        if not focus and lecture.get("learning_objectives"):
            focus = str(lecture.get("learning_objectives", [""])[0])
        if not focus:
            focus = f"Core Topic {idx}"
        title = f"Lecture {idx}: {focus}"
    return title




def _title_needs_focus(title: str) -> bool:
    if not title:
        return True
    t = title.strip().lower()
    if "core topic" in t or "continues the syllabus progression" in t:
        return True
    if re.match(r"^lecture\s*\d+\s*[:\-–—]?\s*$", t):
        return True
    return False


def improve_syllabus_titles(payload: SyllabusPayload, tag_names: list[str]) -> SyllabusPayload:
    if not payload.lectures:
        return payload
    if not tag_names:
        tag_names = []
    improved = []
    for idx, lecture in enumerate(payload.lectures, start=1):
        item = lecture.model_dump() if hasattr(lecture, "model_dump") else dict(lecture)
        title = str(item.get("title", "")).strip()
        if _title_needs_focus(title):
            if tag_names:
                tag = tag_names[(idx - 1) % len(tag_names)]
                item["title"] = f"Lecture {idx}: {tag}"
            else:
                item["title"] = f"Lecture {idx}: Advanced Topic {idx}"
        improved.append(item)
    return SyllabusPayload.model_validate({"lectures": improved})




def _is_generic_summary(summary: str) -> bool:
    if not summary:
        return True
    s = summary.strip().lower()
    return s in {"overview of the topic and its context.", "overview of the topic with focus on core concepts."} or "continues the syllabus progression" in s


def _is_placeholder_objective(obj: str) -> bool:
    if not obj:
        return True
    o = obj.strip().lower()
    return o in {"extend the prior lecture", "deepen conceptual understanding", "connect sources"}


def _is_placeholder_key_terms(terms: list[str]) -> bool:
    if not terms:
        return True
    lowered = [t.strip().lower() for t in terms if t]
    return all(t in {"term1", "term2", "term3", "term4", "term5"} for t in lowered)


def _lecture_is_placeholder(lecture: dict) -> bool:
    summary = str(lecture.get("summary", "")).strip()
    if _is_generic_summary(summary):
        return True
    objectives = lecture.get("learning_objectives") or []
    if objectives and all(_is_placeholder_objective(str(o)) for o in objectives):
        return True
    key_terms = lecture.get("key_terms") or []
    if _is_placeholder_key_terms([str(t) for t in key_terms]):
        return True
    sources = lecture.get("recommended_sources") or []
    if not sources:
        return True
    return False


def _syllabus_has_placeholders(payload: SyllabusPayload) -> bool:
    for lecture in payload.lectures:
        item = lecture.model_dump() if hasattr(lecture, "model_dump") else dict(lecture)
        if _lecture_is_placeholder(item):
            return True
    return False


def _repair_syllabus_placeholders(
    llm: LLMClient,
    context: str,
    tags_data: dict,
    tag_weights: list[TagWeight],
    count: int,
    syllabus: SyllabusPayload,
) -> SyllabusPayload:
    tag_names = [t.get("name") for t in (tags_data.get("tags") or [])]
    weight_spec = [{"name": tw.name, "weight": tw.weight} for tw in tag_weights]
    current = syllabus.model_dump()
    prompt = (
        "You are repairing a syllabus JSON that contains placeholder entries. "
        "Replace any lecture that has generic summaries, placeholder objectives, "
        "placeholder key terms (term1..term5), or empty recommended_sources. "
        "Keep good lectures unchanged. Ensure each lecture is specific to the provided tags "
        "and grounded in the source context. Provide 1-3 concrete recommended_sources per lecture. "
        f"Return exactly {count} lectures.\n\n"
        f"Available tags: {tag_names}\n"
        f"Tag weights: {weight_spec}\n\n"
        f"Current syllabus JSON:\n{json.dumps(current, ensure_ascii=True)}"
    )
    repaired = llm.complete_json(SyllabusPayload, prompt, context)
    return repaired

def _regenerate_syllabus_lecture(
    llm: LLMClient,
    project_id: str,
    lecture: dict,
    tag_weights: list[TagWeight],
) -> dict:
    focus = [TagWeight(name=t.get("name"), weight=t.get("weight", 1.0)) for t in (lecture.get("focus_tags") or [])]
    if not focus:
        focus = tag_weights
    chunks = retrieve_relevant_chunks(project_id, focus, limit=8)
    context_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")[:1500]
        url = chunk.get("url", "")
        context_parts.append(f"[Source {idx}] {url}\n{text}")
    context = "\n\n".join(context_parts)
    prompt = (
        "Generate a single syllabus lecture entry as strict JSON."
        " It must include: lecture_number, title, summary, learning_objectives (3-7),"
        " key_terms (5-15), recommended_sources (1-3 with url and rationale),"
        " estimated_reading_time_minutes, focus_tags."
        f"\nLecture number: {lecture.get('lecture_number')}\n"
        f"Focus tags: {lecture.get('focus_tags', [])}\n"
    )
    generated = llm.complete_json(SyllabusLecture, prompt, context)
    item = generated.model_dump() if hasattr(generated, "model_dump") else dict(generated)
    return item



def apply_weighted_focus(payload: SyllabusPayload, tag_weights: list[TagWeight]) -> SyllabusPayload:
    if not payload.lectures or not tag_weights:
        return payload
    weights = [max(0.0, tw.weight) for tw in tag_weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        return payload
    lecture_count = len(payload.lectures)
    # Largest remainder method for allocation
    exact = [w / total_weight * lecture_count for w in weights]
    counts = [int(x) for x in exact]
    remainder = lecture_count - sum(counts)
    order = sorted(range(len(exact)), key=lambda i: (exact[i] - counts[i]), reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    # Build deterministic tag list
    expanded = []
    for tw, count in zip(tag_weights, counts):
        expanded.extend([tw] * count)
    if not expanded:
        return payload
    improved = []
    for idx, lecture in enumerate(payload.lectures, start=1):
        item = lecture.model_dump() if hasattr(lecture, "model_dump") else dict(lecture)
        primary = expanded[(idx - 1) % len(expanded)]
        # Add a secondary tag for framing if we have more than one tag.
        secondary = None
        if len(tag_weights) > 1:
            secondary = tag_weights[(idx) % len(tag_weights)]
        focus = [{"name": primary.name, "weight": float(primary.weight)}]
        if secondary and secondary.name != primary.name:
            focus.append({"name": secondary.name, "weight": float(secondary.weight)})
        item["focus_tags"] = focus
        title = str(item.get("title", "")).strip()
        if _title_needs_focus(title):
            item["title"] = f"Lecture {idx}: {primary.name}"
        summary = str(item.get("summary", "")).strip()
        if _is_generic_summary(summary):
            item["summary"] = f"Focus on {primary.name}, framed by its role in the overall syllabus.".strip()
        improved.append(item)
    return SyllabusPayload.model_validate({"lectures": improved})
def _coerce_syllabus_payload(data: object, lecture_count: int) -> SyllabusPayload:
    if isinstance(data, dict):
        if "lectures" not in data and "syllabus" in data:
            data = {"lectures": data["syllabus"]}
    if isinstance(data, list):
        data = {"lectures": data}
    if isinstance(data, dict):
        lectures = data.get("lectures", [])
        if lectures and isinstance(lectures[0], dict):
            normalized = []
            for idx, lecture in enumerate(lectures, start=1):
                lecture = dict(lecture)
                lecture.setdefault("lecture_number", idx)
                lecture["title"] = _ensure_title(idx, lecture)
                lecture.setdefault("summary", "Overview of the topic and its context.")
                lecture.setdefault(
                    "learning_objectives",
                    ["Introduce core concepts", "Analyze key sources", "Apply concepts"],
                )
                lecture.setdefault(
                    "key_terms",
                    ["term1", "term2", "term3", "term4", "term5"],
                )
                lecture.setdefault("recommended_sources", [])
                lecture.setdefault("estimated_reading_time_minutes", 60)
                lecture.setdefault("focus_tags", [])
                normalized.append(lecture)
            data = {"lectures": normalized}
    payload = SyllabusPayload.model_validate(data)
    # Normalize lecture titles to be descriptive
    normalized = []
    for idx, lecture in enumerate(payload.lectures, start=1):
        item = lecture.model_dump() if hasattr(lecture, "model_dump") else dict(lecture)
        item["title"] = _ensure_title(idx, item)
        normalized.append(item)
    payload = SyllabusPayload.model_validate({"lectures": normalized})
    if len(payload.lectures) < lecture_count:
        padded = payload.lectures.copy()
        for idx in range(len(padded) + 1, lecture_count + 1):
            padded.append(
                {
                    "lecture_number": idx,
                    "title": _ensure_title(idx, {}),
                    "summary": "Continues the syllabus progression with advanced coverage.",
                    "learning_objectives": [
                        "Extend the prior lecture",
                        "Deepen conceptual understanding",
                        "Connect sources",
                    ],
                    "key_terms": ["term1", "term2", "term3", "term4", "term5"],
                    "recommended_sources": [],
                    "estimated_reading_time_minutes": 60,
                    "focus_tags": [],
                }
            )
        payload = SyllabusPayload.model_validate(
            {"lectures": [l.model_dump() if hasattr(l, "model_dump") else l for l in padded]}
        )
    if len(payload.lectures) > lecture_count:
        payload = SyllabusPayload.model_validate(
            {"lectures": [l.model_dump() if hasattr(l, "model_dump") else l for l in payload.lectures[:lecture_count]]}
        )
    return payload


def _repair_json_with_llm(text: str) -> str | None:
    try:
        llm = get_llm_client()
    except Exception:
        return None
    if llm.name != "openai":
        return None
    prompt = (
        "Fix the JSON below. Return ONLY valid JSON with no extra text. "
        "Preserve as much content as possible.\n\n"
        f"JSON:\n{text}"
    )
    try:
        repaired = llm.complete_markdown(prompt, "")
    except Exception:
        return None
    if not repaired:
        return None
    block = _extract_json_block(repaired) or repaired
    return block


def _parse_json_output(text: str, schema, lecture_count: int | None = None) -> object:
    payload = _extract_json_block(text) or text
    payload = _clean_json(payload)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        repaired = _repair_json_with_llm(payload)
        if repaired:
            data = json.loads(_clean_json(repaired))
        else:
            raise
    if schema is TagsPayload:
        return _coerce_tags_payload(data)
    if schema is SyllabusPayload:
        return _coerce_syllabus_payload(data, lecture_count or settings.syllabus_lecture_count)
    return schema.model_validate(data)


def generate_tags(project_id: str, llm: LLMClient) -> TagsPayload:
    context = _sample_context(project_id)
    if not context.strip():
        context = _sources_context(project_id)
    if not context.strip():
        raise ValueError("No source content available to generate tags. Save sources and run scrape to build the repo.")
    if llm.name == "mock" and not settings.allow_mock_fallback and not settings.test_mode:
        raise ValueError("LLM provider is mock; set OPENAI_API_KEY or llm_provider=openai")
    used_llm = llm
    try:
        pipeline = get_generation_pipeline(llm)
        result = pipeline.generate_tags(context)
        tags = _parse_json_output(str(result.output), TagsPayload)
    except Exception:
        if not settings.allow_mock_fallback and llm.name != "mock":
            raise
        used_llm = MockLLMClient()
        fallback = get_generation_pipeline(used_llm)
        result = fallback.generate_tags(context)
        tags = _parse_json_output(str(result.output), TagsPayload)
    atomic_write_json(project_dir(project_id) / "tags.json", tags.model_dump())
    model_name = settings.openai_model if used_llm.name == "openai" else used_llm.name
    save_run(project_id, "tags_generate", result.prompt, tags.model_dump(), model_name, result.meta)
    return tags


def _fallback_syllabus(lecture_count: int) -> SyllabusPayload:
    lectures = []
    for idx in range(1, lecture_count + 1):
        lectures.append(
            {
                "lecture_number": idx,
                "title": _ensure_title(idx, {}),
                "summary": "Overview of the topic with focus on core concepts.",
                "learning_objectives": [
                    "Explain key concepts",
                    "Connect ideas across sources",
                    "Apply concepts in context",
                ],
                "key_terms": ["term1", "term2", "term3", "term4", "term5"],
                "recommended_sources": [],
                "estimated_reading_time_minutes": 60,
                "focus_tags": [],
            }
        )
    return SyllabusPayload.model_validate({"lectures": lectures})


def generate_syllabus(project_id: str, tag_weights: list[TagWeight], llm: LLMClient, lecture_count: int | None = None) -> SyllabusPayload:
    tags_data = read_json(project_dir(project_id) / "tags.json") or {}
    context = _build_syllabus_context(project_id, tag_weights)
    if not context.strip():
        raise ValueError("No source content available to generate syllabus")
    if llm.name == "mock" and not settings.allow_mock_fallback and not settings.test_mode:
        raise ValueError("LLM provider is mock; set OPENAI_API_KEY or llm_provider=openai")
    used_llm = llm
    count = lecture_count or settings.syllabus_lecture_count
    try:
        pipeline = get_generation_pipeline(llm)
        result = pipeline.generate_syllabus(context, tags_data, tag_weights, count)
        syllabus = _parse_json_output(str(result.output), SyllabusPayload, count)
    except Exception:
        if not settings.allow_mock_fallback and llm.name != "mock":
            raise
        try:
            used_llm = MockLLMClient()
            fallback = get_generation_pipeline(used_llm)
            result = fallback.generate_syllabus(context, tags_data, tag_weights, count)
            syllabus = _parse_json_output(str(result.output), SyllabusPayload, count)
        except Exception:
            result = PipelineResult(output="", prompt="fallback", meta={"pipeline": "fallback"})
            syllabus = _fallback_syllabus(count)
    syllabus = _coerce_syllabus_payload(syllabus.model_dump(), count)
    if llm.name == "openai" and _syllabus_has_placeholders(syllabus):
        syllabus = _repair_syllabus_placeholders(llm, context, tags_data, tag_weights, count, syllabus)
        syllabus = _coerce_syllabus_payload(syllabus.model_dump(), count)
        if _syllabus_has_placeholders(syllabus):
            repaired = []
            for lecture in syllabus.lectures:
                item = lecture.model_dump() if hasattr(lecture, "model_dump") else dict(lecture)
                if _lecture_is_placeholder(item):
                    item = _regenerate_syllabus_lecture(llm, project_id, item, tag_weights)
                repaired.append(item)
            syllabus = _coerce_syllabus_payload({"lectures": repaired}, count)
            if _syllabus_has_placeholders(syllabus):
                raise ValueError("Syllabus contains placeholder lectures after targeted regeneration")
    syllabus = apply_weighted_focus(syllabus, tag_weights)
    tag_names = [t.name for t in tag_weights] if tag_weights else []
    syllabus = improve_syllabus_titles(syllabus, tag_names)
    atomic_write_json(project_dir(project_id) / "syllabus_draft.json", syllabus.model_dump())
    model_name = settings.openai_model if used_llm.name == "openai" else used_llm.name
    save_run(project_id, "syllabus_generate", result.prompt, syllabus.model_dump(), model_name, result.meta)
    return syllabus


def _direct_lecture_fallback(lecture: dict, context: str, llm: LLMClient) -> str:
    criteria = (
        "Maintain consistent lecture structure and length. "
        "Avoid redundancy with prior lectures. Use academic tone, clear definitions, "
        "and coherent transitions. No long verbatim quotes; paraphrase and cite sources via URLs. "
        "Write long-form academic prose only: full paragraphs, no bullet points, no lists. "
        f"Target {settings.lecture_min_words}-{settings.lecture_max_words} words."
    )
    structure = (
        "Required sections in order: Title, Overview, Main Lecture, Glossary, Reading List, Reflection. "
        "Each section must be prose paragraphs (no bullets)."
    )
    spec = json.dumps(lecture, ensure_ascii=True)
    prompt = (
        f"{criteria}\n{structure}\n"
        f"Lecture spec: {spec}\n"
        "Write the full lecture now. Do not stop early."
    )
    return llm.complete_markdown(prompt, context)


def _append_to_main_lecture(markdown: str, addition: str) -> str:
    if not addition.strip():
        return markdown
    if "## Main Lecture" not in markdown:
        return f"{markdown.rstrip()}\n\n## Main Lecture\n\n{addition.strip()}\n"
    parts = markdown.split("## Main Lecture", 1)
    before = parts[0]
    rest = parts[1]
    if "## Glossary" in rest:
        main_body, tail = rest.split("## Glossary", 1)
        main_body = main_body.rstrip() + "\n\n" + addition.strip() + "\n\n"
        return f"{before}## Main Lecture{main_body}## Glossary{tail}"
    return f"{before}## Main Lecture{rest.rstrip()}\n\n{addition.strip()}\n"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _expand_to_min_words(markdown: str, lecture: dict, context: str, llm: LLMClient) -> str:
    target_min = settings.lecture_min_words
    if _word_count(markdown) >= target_min:
        return markdown
    criteria = (
        "Maintain consistent lecture structure and length. "
        "Avoid redundancy with prior lectures. Use academic tone, clear definitions, "
        "and coherent transitions. No long verbatim quotes; paraphrase and cite sources via URLs. "
        "Write long-form academic prose only: full paragraphs, no bullet points, no lists."
    )
    spec = json.dumps(lecture, ensure_ascii=True)
    last_count = _word_count(markdown)
    for _ in range(max(1, settings.lecture_max_expansion_rounds)):
        if last_count >= target_min:
            break
        deficit = target_min - last_count
        prompt = (
            f"{criteria}\n"
            "Write ONLY additional prose paragraphs to extend the Main Lecture section. "
            "Do not include headings or lists. Do not repeat earlier content. "
            f"Add at least {min(deficit, 1200)} words.\n"
            f"Lecture spec: {spec}"
        )
        try:
            addition = llm.complete_markdown(prompt, context)
        except Exception:
            addition = ""
        if not addition or not addition.strip():
            break
        markdown = _append_to_main_lecture(markdown, addition)
        new_count = _word_count(markdown)
        if new_count <= last_count:
            break
        last_count = new_count
    return markdown
    criteria = (
        "Maintain consistent lecture structure and length. "
        "Avoid redundancy with prior lectures. Use academic tone, clear definitions, "
        "and coherent transitions. No long verbatim quotes; paraphrase and cite sources via URLs. "
        "Write long-form academic prose only: full paragraphs, no bullet points, no lists. "
        f"Target {settings.lecture_min_words}-{settings.lecture_max_words} words."
    )
    structure = (
        "Required sections in order: Title, Overview, Main Lecture, Glossary, Reading List, Reflection. "
        "Each section must be prose paragraphs (no bullets)."
    )
    for _ in range(max(1, settings.lecture_max_expansion_rounds)):
        if _word_count(markdown) >= target_min:
            break
        prompt = (
            f"{criteria}\n{structure}\n"
            "Expand the lecture below by adding academic prose to the Main Lecture section. "
            "Do not remove existing content. Keep headings and prose only. "
            f"Add at least {target_min - _word_count(markdown)} words.\n\n"
            f"Current lecture:\n{markdown}"
        )
        try:
            expanded = llm.complete_markdown(prompt, context)
        except Exception:
            expanded = ""
        if expanded and expanded.strip():
            markdown = expanded
    return markdown


def generate_lectures(project_id: str, tag_weights: list[TagWeight], llm: LLMClient, progress_callback: Callable[[int, int, str], None] | None = None) -> list[Path]:
    if llm.name == "mock" and not settings.allow_mock_fallback and not settings.test_mode:
        raise ValueError("LLM provider is mock; set OPENAI_API_KEY or llm_provider=openai")
    syllabus = read_json(project_dir(project_id) / "syllabus_approved.json")
    if not syllabus:
        raise ValueError("Syllabus not approved")
    lectures = syllabus.get("lectures", [])
    outputs: list[Path] = []
    total = len(lectures)
    for idx, lecture in enumerate(lectures, start=1):
        if progress_callback:
            title = lecture.get("title") or f"Lecture {lecture.get('lecture_number', idx)}"
            progress_callback(idx, total, f"Generating lecture {idx}/{total}: {title}")
        try:
            focus = [
                TagWeight(name=t.get("name"), weight=t.get("weight", 1.0))
                for t in lecture.get("focus_tags", [])
            ]
            if not focus:
                focus = tag_weights
            chunks = retrieve_relevant_chunks(project_id, focus, limit=8)
            context_parts = []
            for c_idx, chunk in enumerate(chunks, start=1):
                context_parts.append(
                    f"[Source {c_idx}] {chunk.get('url','')}\n{chunk.get('text','')[:1500]}"
                )
            context = "\n\n".join(context_parts)
            used_llm = llm
            try:
                pipeline = get_generation_pipeline(llm)
                result = pipeline.generate_lecture(lecture, context)
                markdown = result.output
            except Exception:
                if not settings.allow_mock_fallback and llm.name != "mock":
                    markdown = _direct_lecture_fallback(lecture, context, llm)
                    result = PipelineResult(
                        output=markdown,
                        prompt="direct_fallback",
                        meta={"pipeline": "direct_fallback"},
                    )
                else:
                    used_llm = MockLLMClient()
                    fallback = get_generation_pipeline(used_llm)
                    result = fallback.generate_lecture(lecture, context)
                    markdown = result.output
            if not str(markdown).strip():
                markdown = _direct_lecture_fallback(lecture, context, llm)
            markdown = _expand_to_min_words(markdown, lecture, context, llm)
            model_name = settings.openai_model if used_llm.name == "openai" else used_llm.name
            footer = (
                "\n---\n"
                f"Generated on {iso_now()} with {model_name} for project {project_id}.\n"
            )
            markdown = f"{markdown.rstrip()}\n{footer}"
            lecture_title = lecture.get("title") or f"Lecture {lecture.get("lecture_number", idx)}"
            filename = f"{lecture['lecture_number']:02d}-{slugify(lecture_title)}.md"
            path = project_dir(project_id) / "outputs" / "lectures" / filename
            path.write_text(markdown, encoding="utf-8")
            meta = {"lecture": lecture}
            meta.update(result.meta)
            save_run(project_id, "lecture_generate", result.prompt, markdown, model_name, meta)
            outputs.append(path)
        except Exception:
            fallback_text = _direct_lecture_fallback(lecture, "", llm)
            lecture_title = lecture.get("title") or f"Lecture {lecture.get("lecture_number", idx)}"
            filename = f"{lecture['lecture_number']:02d}-{slugify(lecture_title)}.md"
            path = project_dir(project_id) / "outputs" / "lectures" / filename
            footer = (
                "\n---\n"
                f"Generated on {iso_now()} with {llm.name} for project {project_id}.\n"
            )
            path.write_text(f"{fallback_text.rstrip()}\n{footer}", encoding="utf-8")
            outputs.append(path)
        if progress_callback:
            title = lecture.get("title") or f"Lecture {lecture.get('lecture_number', idx)}"
            progress_callback(idx, total, f"Lecture {idx}/{total} generated: {title}")
    return outputs


def generate_essay(project_id: str, topic: str, tag_weights: list[TagWeight], llm: LLMClient) -> Path:
    chunks = retrieve_relevant_chunks(project_id, tag_weights, limit=10)
    context_parts = []
    for idx, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Source {idx}] {chunk.get('url','')}\n{chunk.get('text','')[:1500]}")
    context = "\n\n".join(context_parts)
    used_llm = llm
    try:
        pipeline = get_generation_pipeline(llm)
        result = pipeline.generate_essay(topic, context)
        markdown = result.output
    except Exception:
        if not settings.allow_mock_fallback and llm.name != "mock":
            raise
        used_llm = MockLLMClient()
        fallback = get_generation_pipeline(used_llm)
        result = fallback.generate_essay(topic, context)
        markdown = result.output
    model_name = settings.openai_model if used_llm.name == "openai" else used_llm.name
    footer = (
        "\n---\n"
        f"Generated on {iso_now()} with {model_name} for project {project_id}.\n"
    )
    markdown = f"{markdown.rstrip()}\n{footer}"
    filename = f"essay-{slugify(topic)[:40]}-{settings.app_name[:3].lower()}.md"
    path = project_dir(project_id) / "outputs" / "essays" / filename
    path.write_text(markdown, encoding="utf-8")
    meta = {"topic": topic}
    meta.update(result.meta)
    save_run(project_id, "essay_generate", result.prompt, markdown, model_name, meta)
    return path
