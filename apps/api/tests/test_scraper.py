from __future__ import annotations

from pathlib import Path

from app.scraper import extract_content


def test_extract_content_fallback() -> None:
    html = Path(__file__).parent / "fixtures" / "sample.html"
    content = extract_content(html.read_text(encoding="utf-8"), "https://example.com")
    assert "sample paragraph" in content["text"].lower()
    assert content["title"] == "Sample Page"
