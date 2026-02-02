from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from .config import settings
from .llm import LLMClient, MockLLMClient
from .schemas import SyllabusPayload, TagsPayload, TagWeight


@dataclass
class PipelineResult:
    output: str
    prompt: str
    meta: dict[str, Any]


TAGS_CRITERIA = (
    "Generate a concise, academically oriented topic taxonomy. "
    "No overlap or duplicates. Exactly 12 tags. "
    "Return strict JSON only, no prose."
)

SYLLABUS_CRITERIA = (
    "Ensure a rigorous progression from fundamentals to advanced topics. "
    "Avoid redundancy across lectures. Each lecture should be distinct and build "
    "on prior content. Maintain consistent scope and estimated reading time. "
    "Return strict JSON only, no prose."
)

LECTURE_CRITERIA = (
    "Maintain consistent lecture structure and length. "
    "Avoid redundancy with prior lectures. Use academic tone, clear definitions, "
    "and coherent transitions. No long verbatim quotes; paraphrase and cite sources via URLs. "
    "Write long-form academic prose only: full paragraphs, no bullet points, no lists. "
    "Target 6000-9000 words and do not stop early."
)

ESSAY_CRITERIA = (
    "Academic tone, coherent argumentation, and concise sections. "
    "Avoid redundancy and long verbatim quotes; include attribution links."
)

LECTURE_STRUCTURE = (
    "Required sections in order: Title, Overview, Main Lecture, Glossary, Reading List, "
    "Reflection. Each section must be prose paragraphs (no bullets)."
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\\b\\w+\\b", text))


def _list_item_count(text: str) -> int:
    return len(re.findall(r"^\\s*([-*+]|\\d+\\.)\\s+", text, flags=re.MULTILINE))


def _validate_lecture(markdown: str) -> list[str]:
    issues: list[str] = []
    word_count = _word_count(markdown)
    if word_count < settings.lecture_min_words or word_count > settings.lecture_max_words:
        issues.append(
            f"Ensure length is between {settings.lecture_min_words} and {settings.lecture_max_words} words (currently {word_count})."
        )
    list_items = _list_item_count(markdown)
    if list_items > settings.lecture_max_list_items:
        issues.append(
            f"Reduce list items to at most {settings.lecture_max_list_items} total and convert lists to prose."
        )
    return issues


def _length_instruction(word_count: int) -> str:
    if word_count < settings.lecture_min_words:
        return (
            f"Too short by approximately {settings.lecture_min_words - word_count} words. "
            "Append new paragraphs to the Main Lecture (and, if needed, Overview/Reflection) "
            "without removing existing content. Add academic depth, examples, and citations. "
            f"Target total length between {settings.lecture_min_words} and "
            f"{settings.lecture_max_words} words. Keep all sections as prose."
        )
    if word_count > settings.lecture_max_words:
        return (
            f"Too long by approximately {word_count - settings.lecture_max_words} words. "
            f"Condense the lecture to between {settings.lecture_min_words} and "
            f"{settings.lecture_max_words} words while preserving the required sections."
        )
    return ""


def _ensure_lecture_structure(markdown: str, title: str) -> str:
    required_headings = [
        "# ",
        "## Overview",
        "## Main Lecture",
        "## Glossary",
        "## Reading List",
        "## Reflection",
    ]
    output = markdown.strip()
    if not output.startswith("# "):
        output = f"# {title}\\n\\n{output}"
    for heading in required_headings[1:]:
        if heading not in output:
            output += f"\\n\\n{heading}\\n\\n"
    return output


def _split_terms(terms: list[str], parts: int) -> list[list[str]]:
    if parts <= 1:
        return [terms]
    buckets: list[list[str]] = [[] for _ in range(parts)]
    for idx, term in enumerate(terms):
        buckets[idx % parts].append(term)
    return buckets


def _build_part_focuses(lecture: dict, parts: int) -> list[str]:
    terms = [str(t) for t in lecture.get("key_terms", []) if t]
    objectives = [str(o) for o in lecture.get("learning_objectives", []) if o]
    term_groups = _split_terms(terms, parts)
    focuses: list[str] = []
    for idx in range(parts):
        group_terms = ", ".join(term_groups[idx]) if term_groups[idx] else "core concepts"
        objective = objectives[idx % len(objectives)] if objectives else "advance the topic logically"
        focuses.append(f"{objective}. Key terms: {group_terms}.")
    return focuses


def _ensure_prose(llm: LLMClient, text: str, context: str, lecture_spec_json: str) -> str:
    if _list_item_count(text) <= settings.lecture_max_list_items:
        return text
    prompt = (
        "Rewrite the section as continuous prose paragraphs. "
        "Remove bullet points and lists. Preserve meaning and citations. "
        f"Lecture spec: {lecture_spec_json}"
    )
    revised = llm.complete_markdown(prompt, context)
    return revised or text

def _stable_outline_sections(lecture: dict, parts: int) -> list[dict[str, str]]:
    objectives = [str(o) for o in lecture.get("learning_objectives", []) if o]
    key_terms = [str(t) for t in lecture.get("key_terms", []) if t]
    parts = max(1, parts)
    base_titles = objectives or key_terms
    if not base_titles:
        base_titles = [f"Core Theme {i}" for i in range(1, parts + 1)]
    sections = []
    for idx in range(parts):
        title = base_titles[idx % len(base_titles)]
        focus = objectives[idx % len(objectives)] if objectives else title
        terms = ", ".join(key_terms[idx::parts]) if key_terms else "core concepts"
        sections.append({"title": title, "focus": focus, "terms": terms})
    return sections




def _safe_complete_markdown(llm: LLMClient, prompt: str, context: str) -> str:
    try:
        return llm.complete_markdown(prompt, context)
    except Exception:
        short_context = "\n\n".join(context.split("\n\n")[:2])
        try:
            return llm.complete_markdown(prompt, short_context)
        except Exception:
            return ""



def _generate_chunked_lecture(llm: LLMClient, lecture: dict, context: str) -> str:
    title = lecture.get("title", "Lecture")
    lecture_spec_json = json.dumps(lecture, ensure_ascii=True)
    parts = max(1, settings.lecture_chunk_parts)
    overhead = settings.lecture_chunk_overhead_words
    main_target = max(settings.lecture_min_words, settings.lecture_target_words) - overhead
    target_per_part = max(1200, int(main_target / parts))
    outline_sections = _stable_outline_sections(lecture, parts)

    overview_prompt = (
        f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}\n"
        "Write the Overview section in prose (no bullets). "
        f"Anchor to the lecture spec: {lecture_spec_json}."
    )
    overview = _safe_complete_markdown(llm, overview_prompt, context) or ""
    overview = _ensure_prose(llm, overview, context, lecture_spec_json)
    if _word_count(overview) < 80:
        overview = _safe_complete_markdown(llm, overview_prompt, context) or overview

    main_parts: list[str] = []
    covered: list[str] = []
    for idx, section in enumerate(outline_sections, start=1):
        focus = section["focus"]
        terms = section["terms"]
        coverage = " ".join(covered) if covered else "None yet."
        part_prompt = (
            f"{LECTURE_CRITERIA}\n"
            "Write ONLY the Main Lecture subsection for this part with heading '### {section_title}'. "
            "Use continuous prose paragraphs. No bullets. "
            f"Part {idx}/{parts}. Focus: {focus}. Key terms: {terms}. "
            f"Avoid repeating coverage: {coverage}. "
            f"Target about {target_per_part} words for this part."
        ).format(section_title=section["title"])
        part_text = _safe_complete_markdown(llm, part_prompt, context) or ""
        if not part_text.lstrip().startswith("### "):
            part_text = f"### {section['title']}\n\n{part_text.strip()}"
        part_text = _ensure_prose(llm, part_text, context, lecture_spec_json)
        if _word_count(part_text) < int(target_per_part * 0.6):
            expand_prompt = (
                f"{LECTURE_CRITERIA}\n"
                "Expand the following subsection with additional academic paragraphs, "
                "examples, and citations. Keep prose only. Do not add new headings.\n\n"
                f"{part_text}"
            )
            part_text = _safe_complete_markdown(llm, expand_prompt, context) or part_text
            part_text = _ensure_prose(llm, part_text, context, lecture_spec_json)
        if _word_count(part_text) < 80:
            shortened_context = "\n\n".join(context.split("\n\n")[:2])
            retry_text = _safe_complete_markdown(llm, part_prompt, shortened_context)
            if retry_text:
                part_text = retry_text
            if not part_text.lstrip().startswith("### "):
                part_text = f"### {section['title']}\n\n{part_text.strip()}"
            part_text = _ensure_prose(llm, part_text, shortened_context, lecture_spec_json)
        main_parts.append(part_text.strip())
        covered.append(section["title"])

    glossary_prompt = (
        "Write the Glossary section as prose paragraphs defining the key terms. "
        "No bullet points. Include inline term definitions.\n"
        f"Key terms: {', '.join(lecture.get('key_terms', []))}."
    )
    glossary = _safe_complete_markdown(llm, glossary_prompt, context) or ""
    glossary = _ensure_prose(llm, glossary, context, lecture_spec_json)
    if _word_count(glossary) < 50:
        glossary = _safe_complete_markdown(llm, glossary_prompt, context) or glossary

    reading_prompt = (
        "Write the Reading List section as prose (no bullets). "
        "Mention 4-6 source URLs inline with brief rationales.\n"
        f"Lecture spec: {lecture_spec_json}."
    )
    reading = _safe_complete_markdown(llm, reading_prompt, context) or ""
    reading = _ensure_prose(llm, reading, context, lecture_spec_json)
    if _word_count(reading) < 50:
        reading = _safe_complete_markdown(llm, reading_prompt, context) or reading

    reflection_prompt = (
        "Write the Reflection section as a single cohesive prose paragraph. "
        "No bullets. Tie back to the lecture objectives."
    )
    reflection = _safe_complete_markdown(llm, reflection_prompt, context) or ""
    reflection = _ensure_prose(llm, reflection, context, lecture_spec_json)
    if _word_count(reflection) < 50:
        reflection = _safe_complete_markdown(llm, reflection_prompt, context) or reflection

    main_body = "\n\n".join([p for p in main_parts if p])
    assembled = (
        f"# {title}\n\n"
        f"## Overview\n\n{overview.strip()}\n\n"
        f"## Main Lecture\n\n{main_body.strip()}\n\n"
        f"## Glossary\n\n{glossary.strip()}\n\n"
        f"## Reading List\n\n{reading.strip()}\n\n"
        f"## Reflection\n\n{reflection.strip()}\n"
    )
    if _word_count(assembled) < 200:
        direct_prompt = (
            f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}\n"
            "Write the full lecture now. Do not stop early."
        )
        assembled = _safe_complete_markdown(llm, direct_prompt, context) or assembled
    return assembled


class MockPipeline:
    def __init__(self, llm: MockLLMClient) -> None:
        self.llm = llm

    def generate_tags(self, context: str) -> PipelineResult:
        prompt = "Return JSON with 12 tags."
        tags = self.llm.complete_json(TagsPayload, prompt, context)
        return PipelineResult(
            output=tags.model_dump_json(),
            prompt=prompt,
            meta={"pipeline": "mock"},
        )

    def generate_syllabus(
        self,
        context: str,
        tags: dict,
        tag_weights: list[TagWeight],
        lecture_count: int,
    ) -> PipelineResult:
        prompt = f"Return JSON with {lecture_count} lectures."
        syllabus = self.llm.complete_json(SyllabusPayload, prompt, context)
        return PipelineResult(
            output=syllabus.model_dump_json(),
            prompt=prompt,
            meta={"pipeline": "mock"},
        )

    def generate_lecture(self, lecture: dict, context: str) -> PipelineResult:
        # Apply test mode overrides locally to avoid long outputs during smoke tests.
        if settings.test_mode:
            settings.lecture_min_words = settings.test_lecture_min_words
            settings.lecture_max_words = settings.test_lecture_max_words
            settings.lecture_target_words = settings.test_lecture_target_words
            settings.lecture_word_tolerance = settings.test_lecture_word_tolerance
            settings.lecture_chunk_parts = settings.test_lecture_chunk_parts
            settings.lecture_chunk_overhead_words = settings.test_lecture_chunk_overhead_words
        lecture_spec_json = json.dumps(lecture, ensure_ascii=True)
        prompt = f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}"
        if settings.lecture_chunked_enabled:
            markdown = _generate_chunked_lecture(self.llm, lecture, context)
        else:
            try:
                with self.dspy.context(lm=self.lm):
                    markdown = self.lecture_module(
                        lecture_spec_json=lecture_spec_json,
                        context=context,
                        criteria=LECTURE_CRITERIA,
                        structure=LECTURE_STRUCTURE,
                    )
            except Exception:
                markdown = ""
            if not isinstance(markdown, str):
                markdown = str(markdown or "")
            if _word_count(markdown) < 100:
                direct_prompt = (
                    f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}\n"
                    "Write the full lecture now. Do not stop early."
                )
                markdown = _safe_complete_markdown(self.llm, direct_prompt, context) or markdown
        markdown = _ensure_lecture_structure(markdown, lecture.get("title", "Lecture"))

        # Repair loop: revise and length-adjust until within bounds, never hard-fail.
        issues: list[str] = []
        for _ in range(max(1, settings.lecture_max_expansion_rounds)):
            issues = _validate_lecture(markdown)
            if not issues:
                break
            word_count = _word_count(markdown)
            length_note = _length_instruction(word_count)
            critique = " ".join(issues + ([length_note] if length_note else []))

            try:
                with self.dspy.context(lm=self.lm):
                    revised = self.lecture_module.revise(
                        lecture_markdown=markdown,
                        critique=critique,
                        criteria=LECTURE_CRITERIA,
                        structure=LECTURE_STRUCTURE,
                    )
                markdown = revised.lecture_markdown_revised or markdown
                with self.dspy.context(lm=self.lm):
                    adjusted = self.lecture_module.length_adjust(
                        lecture_markdown=markdown,
                        target_words=settings.lecture_target_words,
                        tolerance=settings.lecture_word_tolerance,
                    )
                if adjusted.lecture_markdown_adjusted:
                    markdown = adjusted.lecture_markdown_adjusted
            except Exception:
                fallback_prompt = (
                    f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}\n"
                    f"{critique}\n\n"
                    f"Lecture:\n{markdown}"
                )
                fallback = _safe_complete_markdown(self.llm, fallback_prompt, context)
                if fallback:
                    markdown = fallback

            if _word_count(markdown) < settings.lecture_min_words:
                expand_prompt = (
                    f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}\n"
                    "Expand the lecture by adding academic prose to the Main Lecture section. "
                    "Do not remove existing content. Keep headings and prose only. "
                    f"Add at least {settings.lecture_min_words - _word_count(markdown)} words."
                )
                expanded = _safe_complete_markdown(self.llm, expand_prompt, context)
                if expanded:
                    markdown = expanded

            markdown = _ensure_lecture_structure(markdown, lecture.get("title", "Lecture"))

        issues = _validate_lecture(markdown)
        return PipelineResult(
            output=markdown,
            prompt=prompt,
            meta={
                "pipeline": "dspy",
                "criteria": LECTURE_CRITERIA,
                "structure": LECTURE_STRUCTURE,
                "word_count": _word_count(markdown),
                "target_words": settings.lecture_target_words,
                "issues": issues,
            },
        )

    def generate_essay(self, topic: str, context: str) -> PipelineResult:
        prompt = "Generate essay markdown."
        markdown = self.llm.complete_markdown(prompt, context)
        return PipelineResult(
            output=markdown,
            prompt=prompt,
            meta={"pipeline": "mock"},
        )


class DSPyPipeline:
    def __init__(self, llm: LLMClient) -> None:
        import dspy  # type: ignore

        self.dspy = dspy
        self.llm = llm
        self.lm = self._build_lm()

        self.tag_module = self._TagModule()
        self.syllabus_module = self._SyllabusModule()
        self.lecture_module = self._LectureModule()
        self.essay_module = self._EssayModule()

    def _build_lm(self):
        dspy = self.dspy
        if self.llm.name != "openai":
            raise ValueError("DSPy pipeline currently supports only OpenAI provider.")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for DSPy OpenAI pipeline.")

        lm = None
        if hasattr(dspy, "OpenAI"):
            lm = dspy.OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
        elif hasattr(dspy, "LM"):
            try:
                lm = dspy.LM(model=settings.openai_model, api_key=settings.openai_api_key)
            except TypeError:
                lm = dspy.LM(f"openai/{settings.openai_model}", api_key=settings.openai_api_key)
        if lm is None:
            raise RuntimeError("No compatible DSPy LM wrapper found.")
        return lm

    class _TagSignature:  # type: ignore[misc]
        def __init__(self, dspy: Any) -> None:
            self.Signature = dspy.Signature
            self.InputField = dspy.InputField
            self.OutputField = dspy.OutputField

        def build(self) -> Any:
            dspy = self

            class TagSignature(dspy.Signature):
                context: str = dspy.InputField(desc="Corpus excerpts.")
                criteria: str = dspy.InputField(desc="Tagging criteria.")
                tags_json: str = dspy.OutputField(
                    desc="Strict JSON with key 'tags' and exactly 12 entries."
                )

            return TagSignature

    class _SyllabusSignature:  # type: ignore[misc]
        def __init__(self, dspy: Any) -> None:
            self.Signature = dspy.Signature
            self.InputField = dspy.InputField
            self.OutputField = dspy.OutputField

        def build(self) -> Any:
            dspy = self

            class SyllabusDraft(dspy.Signature):
                context: str = dspy.InputField(desc="Corpus excerpts.")
                tags_json: str = dspy.InputField(desc="Tags JSON.")
                tag_weights_json: str = dspy.InputField(desc="Tag weights JSON.")
                criteria: str = dspy.InputField(desc="Syllabus criteria.")
                syllabus_json: str = dspy.OutputField(
                    desc=(
                        "Strict JSON with key 'lectures' containing 30 lecture objects. "
                        "Each object must include lecture_number, title, summary, "
                        "learning_objectives, key_terms, recommended_sources, "
                        "estimated_reading_time_minutes, focus_tags."
                    )
                )

            class SyllabusCritique(dspy.Signature):
                syllabus_json: str = dspy.InputField(desc="Draft syllabus JSON.")
                criteria: str = dspy.InputField(desc="Criteria to enforce.")
                critique: str = dspy.OutputField(desc="Actionable critique list.")

            class SyllabusRevise(dspy.Signature):
                syllabus_json: str = dspy.InputField(desc="Draft syllabus JSON.")
                critique: str = dspy.InputField(desc="Critique to address.")
                criteria: str = dspy.InputField(desc="Criteria to enforce.")
                syllabus_json_revised: str = dspy.OutputField(desc="Revised JSON.")

            return SyllabusDraft, SyllabusCritique, SyllabusRevise

    class _LectureSignature:  # type: ignore[misc]
        def __init__(self, dspy: Any) -> None:
            self.Signature = dspy.Signature
            self.InputField = dspy.InputField
            self.OutputField = dspy.OutputField

        def build(self) -> Any:
            dspy = self

            class LectureDraft(dspy.Signature):
                lecture_spec_json: str = dspy.InputField(desc="Lecture spec JSON.")
                context: str = dspy.InputField(desc="Retrieved sources.")
                criteria: str = dspy.InputField(desc="Lecture criteria.")
                structure: str = dspy.InputField(desc="Required section order.")
                lecture_markdown: str = dspy.OutputField(desc="Markdown lecture.")

            class LectureCritique(dspy.Signature):
                lecture_markdown: str = dspy.InputField(desc="Draft lecture.")
                criteria: str = dspy.InputField(desc="Criteria to enforce.")
                critique: str = dspy.OutputField(desc="Actionable critique list.")

            class LectureRevise(dspy.Signature):
                lecture_markdown: str = dspy.InputField(desc="Draft lecture.")
                critique: str = dspy.InputField(desc="Critique to address.")
                criteria: str = dspy.InputField(desc="Criteria to enforce.")
                structure: str = dspy.InputField(desc="Required section order.")
                lecture_markdown_revised: str = dspy.OutputField(desc="Revised lecture.")

            class LectureLengthAdjust(dspy.Signature):
                lecture_markdown: str = dspy.InputField(desc="Draft lecture.")
                target_words: int = dspy.InputField(desc="Target word count.")
                tolerance: float = dspy.InputField(desc="Allowed tolerance ratio.")
                lecture_markdown_adjusted: str = dspy.OutputField(desc="Adjusted lecture.")

            return LectureDraft, LectureCritique, LectureRevise, LectureLengthAdjust

    class _EssaySignature:  # type: ignore[misc]
        def __init__(self, dspy: Any) -> None:
            self.Signature = dspy.Signature
            self.InputField = dspy.InputField
            self.OutputField = dspy.OutputField

        def build(self) -> Any:
            dspy = self

            class EssayDraft(dspy.Signature):
                topic: str = dspy.InputField(desc="Essay topic.")
                context: str = dspy.InputField(desc="Retrieved sources.")
                criteria: str = dspy.InputField(desc="Essay criteria.")
                essay_markdown: str = dspy.OutputField(desc="Markdown essay.")

            class EssayCritique(dspy.Signature):
                essay_markdown: str = dspy.InputField(desc="Draft essay.")
                criteria: str = dspy.InputField(desc="Criteria to enforce.")
                critique: str = dspy.OutputField(desc="Actionable critique list.")

            class EssayRevise(dspy.Signature):
                essay_markdown: str = dspy.InputField(desc="Draft essay.")
                critique: str = dspy.InputField(desc="Critique to address.")
                criteria: str = dspy.InputField(desc="Criteria to enforce.")
                essay_markdown_revised: str = dspy.OutputField(desc="Revised essay.")

            return EssayDraft, EssayCritique, EssayRevise

    class _TagModule:  # type: ignore[misc]
        def __init__(self) -> None:
            import dspy  # type: ignore

            TagSignature = DSPyPipeline._TagSignature(dspy).build()
            self.predict = dspy.Predict(TagSignature)

        def __call__(self, context: str, criteria: str) -> Any:
            return self.predict(context=context, criteria=criteria)

    class _SyllabusModule:  # type: ignore[misc]
        def __init__(self) -> None:
            import dspy  # type: ignore

            SyllabusDraft, SyllabusCritique, SyllabusRevise = DSPyPipeline._SyllabusSignature(dspy).build()
            self.draft = dspy.Predict(SyllabusDraft)
            self.critique = dspy.Predict(SyllabusCritique)
            self.revise = dspy.Predict(SyllabusRevise)

        def __call__(self, context: str, tags_json: str, tag_weights_json: str, criteria: str) -> Any:
            draft = self.draft(
                context=context,
                tags_json=tags_json,
                tag_weights_json=tag_weights_json,
                criteria=criteria,
            )
            critique = self.critique(syllabus_json=draft.syllabus_json, criteria=criteria)
            revised = self.revise(
                syllabus_json=draft.syllabus_json,
                critique=critique.critique,
                criteria=criteria,
            )
            return revised.syllabus_json_revised or draft.syllabus_json

    class _LectureModule:  # type: ignore[misc]
        def __init__(self) -> None:
            import dspy  # type: ignore

            (
                LectureDraft,
                LectureCritique,
                LectureRevise,
                LectureLengthAdjust,
            ) = DSPyPipeline._LectureSignature(dspy).build()
            self.draft = dspy.Predict(LectureDraft)
            self.critique = dspy.Predict(LectureCritique)
            self.revise = dspy.Predict(LectureRevise)
            self.length_adjust = dspy.Predict(LectureLengthAdjust)

        def __call__(self, lecture_spec_json: str, context: str, criteria: str, structure: str) -> str:
            draft = self.draft(
                lecture_spec_json=lecture_spec_json,
                context=context,
                criteria=criteria,
                structure=structure,
            )
            critique = self.critique(lecture_markdown=draft.lecture_markdown, criteria=criteria)
            revised = self.revise(
                lecture_markdown=draft.lecture_markdown,
                critique=critique.critique,
                criteria=criteria,
                structure=structure,
            )
            output = revised.lecture_markdown_revised or draft.lecture_markdown

            words = _word_count(output)
            target = settings.lecture_target_words
            tolerance = settings.lecture_word_tolerance
            lower = int(target * (1 - tolerance))
            upper = int(target * (1 + tolerance))
            if words < lower or words > upper:
                adjusted = self.length_adjust(
                    lecture_markdown=output,
                    target_words=target,
                    tolerance=tolerance,
                )
                if adjusted.lecture_markdown_adjusted:
                    output = adjusted.lecture_markdown_adjusted
            return output

    class _EssayModule:  # type: ignore[misc]
        def __init__(self) -> None:
            import dspy  # type: ignore

            EssayDraft, EssayCritique, EssayRevise = DSPyPipeline._EssaySignature(dspy).build()
            self.draft = dspy.Predict(EssayDraft)
            self.critique = dspy.Predict(EssayCritique)
            self.revise = dspy.Predict(EssayRevise)

        def __call__(self, topic: str, context: str, criteria: str) -> str:
            draft = self.draft(topic=topic, context=context, criteria=criteria)
            critique = self.critique(essay_markdown=draft.essay_markdown, criteria=criteria)
            revised = self.revise(
                essay_markdown=draft.essay_markdown,
                critique=critique.critique,
                criteria=criteria,
            )
            return revised.essay_markdown_revised or draft.essay_markdown

    def generate_tags(self, context: str) -> PipelineResult:
        prompt = f"{TAGS_CRITERIA}"
        with self.dspy.context(lm=self.lm):
            result = self.tag_module(context=context, criteria=TAGS_CRITERIA)
        return PipelineResult(
            output=result.tags_json,
            prompt=prompt,
            meta={"pipeline": "dspy", "criteria": TAGS_CRITERIA},
        )

    def generate_syllabus(
        self,
        context: str,
        tags: dict,
        tag_weights: list[TagWeight],
        lecture_count: int,
    ) -> PipelineResult:
        prompt = (
            f"{SYLLABUS_CRITERIA} "
            f"Return exactly {lecture_count} lectures. "
            "Each lecture must include: lecture_number (1..N), title, summary, "
            "learning_objectives (3-7 bullets), key_terms (5-15), "
            "recommended_sources (list of {url, rationale}), "
            "estimated_reading_time_minutes (int), focus_tags (list of {name, weight})."
        )
        tags_json = json.dumps(tags, ensure_ascii=True)
        tag_weights_json = json.dumps([tw.model_dump() for tw in tag_weights], ensure_ascii=True)
        with self.dspy.context(lm=self.lm):
            result = self.syllabus_module(
                context=context,
                tags_json=tags_json,
                tag_weights_json=tag_weights_json,
                criteria=prompt,
            )
        output = result if isinstance(result, str) else getattr(result, "syllabus_json_revised", None) or getattr(result, "syllabus_json", None) or str(result)
        return PipelineResult(
            output=output,
            prompt=prompt,
            meta={
                "pipeline": "dspy",
                "criteria": SYLLABUS_CRITERIA,
                "lecture_count": lecture_count,
            },
        )

    def generate_lecture(self, lecture: dict, context: str) -> PipelineResult:
        # Apply test mode overrides locally to avoid long outputs during smoke tests.
        if settings.test_mode:
            settings.lecture_min_words = settings.test_lecture_min_words
            settings.lecture_max_words = settings.test_lecture_max_words
            settings.lecture_target_words = settings.test_lecture_target_words
            settings.lecture_word_tolerance = settings.test_lecture_word_tolerance
            settings.lecture_chunk_parts = settings.test_lecture_chunk_parts
            settings.lecture_chunk_overhead_words = settings.test_lecture_chunk_overhead_words
        lecture_spec_json = json.dumps(lecture, ensure_ascii=True)
        prompt = f"{LECTURE_CRITERIA}\\n{LECTURE_STRUCTURE}"
        if settings.lecture_chunked_enabled:
            markdown = _generate_chunked_lecture(self.llm, lecture, context)
        else:
            with self.dspy.context(lm=self.lm):
                markdown = self.lecture_module(
                    lecture_spec_json=lecture_spec_json,
                    context=context,
                    criteria=LECTURE_CRITERIA,
                    structure=LECTURE_STRUCTURE,
                )
            if not isinstance(markdown, str):
                markdown = str(markdown or "")
            if _word_count(markdown) < 100:
                direct_prompt = (
                    f"{LECTURE_CRITERIA}\\n{LECTURE_STRUCTURE}\\n"
                    "Write the full lecture now. Do not stop early."
                )
                markdown = self.llm.complete_markdown(direct_prompt, context)
        markdown = _ensure_lecture_structure(markdown, lecture.get("title", "Lecture"))

        for _ in range(max(1, settings.dspy_max_revisions)):
            issues = _validate_lecture(markdown)
            if not issues:
                break
            critique = " ".join(issues)
            try:
                with self.dspy.context(lm=self.lm):
                    revised = self.lecture_module.revise(
                        lecture_markdown=markdown,
                        critique=critique,
                        criteria=LECTURE_CRITERIA,
                        structure=LECTURE_STRUCTURE,
                    )
                markdown = revised.lecture_markdown_revised or markdown
                with self.dspy.context(lm=self.lm):
                    adjusted = self.lecture_module.length_adjust(
                        lecture_markdown=markdown,
                        target_words=settings.lecture_target_words,
                        tolerance=settings.lecture_word_tolerance,
                    )
                if adjusted.lecture_markdown_adjusted:
                    markdown = adjusted.lecture_markdown_adjusted
            except Exception:
                fallback_prompt = (
                    f"{LECTURE_CRITERIA}\n{LECTURE_STRUCTURE}\n"
                    f"{critique}\n\n"
                    f"Lecture:\n{markdown}"
                )
                fallback = _safe_complete_markdown(self.llm, fallback_prompt, context)
                if fallback:
                    markdown = fallback

        issues = _validate_lecture(markdown)
        if issues and any("Ensure length" in issue for issue in issues):
            for _ in range(settings.lecture_max_expansion_rounds):
                word_count = _word_count(markdown)
                if settings.lecture_min_words <= word_count <= settings.lecture_max_words:
                    break
                length_note = _length_instruction(word_count)
                critique = " ".join(issues + ([length_note] if length_note else []))
                with self.dspy.context(lm=self.lm):
                    revised = self.lecture_module.revise(
                        lecture_markdown=markdown,
                        critique=critique,
                        criteria=LECTURE_CRITERIA,
                        structure=LECTURE_STRUCTURE,
                    )
                markdown = revised.lecture_markdown_revised or markdown
                with self.dspy.context(lm=self.lm):
                    adjusted = self.lecture_module.length_adjust(
                        lecture_markdown=markdown,
                        target_words=settings.lecture_target_words,
                        tolerance=settings.lecture_word_tolerance,
                    )
                if adjusted.lecture_markdown_adjusted:
                    markdown = adjusted.lecture_markdown_adjusted
                issues = _validate_lecture(markdown)
                if not issues:
                    break

        issues = _validate_lecture(markdown)
        if issues:
            raise ValueError("Lecture validation failed: " + " ".join(issues))

        return PipelineResult(
            output=markdown,
            prompt=prompt,
            meta={
                "pipeline": "dspy",
                "criteria": LECTURE_CRITERIA,
                "structure": LECTURE_STRUCTURE,
                "word_count": _word_count(markdown),
                "target_words": settings.lecture_target_words,
                "issues": issues,
            },
        )

    def generate_essay(self, topic: str, context: str) -> PipelineResult:
        prompt = f"{ESSAY_CRITERIA}"
        with self.dspy.context(lm=self.lm):
            markdown = self.essay_module(topic=topic, context=context, criteria=ESSAY_CRITERIA)
        return PipelineResult(
            output=markdown,
            prompt=prompt,
            meta={"pipeline": "dspy", "criteria": ESSAY_CRITERIA},
        )


def get_generation_pipeline(llm: LLMClient) -> MockPipeline | DSPyPipeline:
    if not settings.dspy_enabled or llm.name == "mock":
        if not isinstance(llm, MockLLMClient):
            llm = MockLLMClient()
        return MockPipeline(llm)
    return DSPyPipeline(llm)
