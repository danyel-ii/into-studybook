from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from .config import settings

try:
    import lxml  # type: ignore

    HAS_LXML = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LXML = False


def _html_parser() -> str:
    return "lxml" if HAS_LXML else "html.parser"


def _xml_parser() -> str:
    return "xml" if HAS_LXML else "html.parser"
from .schemas import Source
from .storage import append_jsonl, atomic_write_json, project_dir, read_json

try:
    import trafilatura
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return f"https://{url}"
    return url


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_time: dict[str, float] = {}

    async def wait(self, host: str) -> None:
        lock = self._locks[host]
        async with lock:
            now = asyncio.get_event_loop().time()
            last = self._last_time.get(host, 0.0)
            wait_for = self.min_interval - (now - last)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_time[host] = asyncio.get_event_loop().time()


class RobotsCache:
    def __init__(self) -> None:
        self._parsers: dict[str, RobotFileParser] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._parsers:
            robots_url = urljoin(base, "/robots.txt")
            parser = RobotFileParser()
            try:
                resp = await client.get(robots_url, headers={"User-Agent": settings.user_agent})
                if resp.status_code < 400:
                    parser.parse(resp.text.splitlines())
                else:
                    parser.allow_all = True
            except Exception:
                parser.allow_all = True
            self._parsers[base] = parser
        parser = self._parsers[base]
        return parser.can_fetch(settings.user_agent, url)


def _link_ok(url: str, source: Source) -> bool:
    if source.include_patterns:
        if not any(re.search(p, url) for p in source.include_patterns):
            return False
    if source.exclude_patterns:
        if any(re.search(p, url) for p in source.exclude_patterns):
            return False
    return True


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, _html_parser())
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        if not href:
            continue
        absolute = urljoin(base_url, href)
        links.append(absolute)
    return links


def _same_origin(base: str, target: str) -> bool:
    base_p = urlparse(base)
    target_p = urlparse(target)
    return base_p.scheme == target_p.scheme and base_p.netloc == target_p.netloc


def _within_path(base: str, target: str) -> bool:
    base_p = urlparse(base)
    target_p = urlparse(target)
    return target_p.path.startswith(base_p.path)


def _parse_sitemap(xml_text: str) -> list[str]:
    soup = BeautifulSoup(xml_text, _xml_parser())
    urls = [loc.text for loc in soup.find_all("loc") if loc.text]
    return urls


def extract_content(html: str, url: str) -> dict[str, Any]:
    if trafilatura is not None:
        try:
            data = trafilatura.bare_extraction(html, url=url)
            if data and data.get("text"):
                return {
                    "text": data.get("text", ""),
                    "title": data.get("title") or "",
                    "author": data.get("author") or "",
                    "published_at": data.get("date") or "",
                }
        except Exception:
            pass
    soup = BeautifulSoup(html, _html_parser())
    title = soup.title.text.strip() if soup.title and soup.title.text else ""
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    text = " ".join(soup.get_text(" ").split())
    return {"text": text, "title": title, "author": "", "published_at": ""}


def _cache_paths(project_id: str, url: str) -> tuple[dict, str, str]:
    cache_dir = project_dir(project_id) / "cache"
    cache_index_path = cache_dir / "index.json"
    cache_index = read_json(cache_index_path, default={})
    url_hash = _hash_url(url)
    body_path = str(cache_dir / f"{url_hash}.html")
    return cache_index, str(cache_index_path), body_path


async def fetch_url(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    url: str,
    project_id: str,
) -> tuple[int, str, dict[str, str]]:
    parsed = urlparse(url)
    await limiter.wait(parsed.netloc)

    cache_index, cache_index_path, body_path = _cache_paths(project_id, url)
    cache_entry = cache_index.get(url, {})
    headers = {"User-Agent": settings.user_agent}
    if cache_entry.get("etag"):
        headers["If-None-Match"] = cache_entry["etag"]
    if cache_entry.get("last_modified"):
        headers["If-Modified-Since"] = cache_entry["last_modified"]

    last_status = 0
    for attempt in range(settings.max_retries + 1):
        try:
            resp = await client.get(url, headers=headers, follow_redirects=True)
        except httpx.RequestError:
            last_status = 0
            if attempt < settings.max_retries:
                await asyncio.sleep(2**attempt)
                continue
            return 0, "", {"cached": "false"}

        last_status = resp.status_code
        if resp.status_code == 304 and cache_entry.get("body_path"):
            try:
                with open(cache_entry["body_path"], "r", encoding="utf-8") as f:
                    return 200, f.read(), {"cached": "true"}
            except Exception:
                pass

        if resp.status_code < 400:
            body = resp.text
            with open(body_path, "w", encoding="utf-8") as f:
                f.write(body)
            cache_index[url] = {
                "etag": resp.headers.get("ETag"),
                "last_modified": resp.headers.get("Last-Modified"),
                "body_path": body_path,
                "fetched_at": _now(),
            }
            atomic_write_json(Path(cache_index_path), cache_index)
            return resp.status_code, body, {"cached": "false"}

        if resp.status_code in {429, 500, 502, 503, 504} and attempt < settings.max_retries:
            await asyncio.sleep(2**attempt)
            continue

        break

    return last_status, "", {"cached": "false"}


async def discover_urls(client: httpx.AsyncClient, source: Source) -> list[str]:
    url = normalize_url(str(source.url))
    if source.mode == "single":
        return [url]

    if source.mode == "sitemap":
        sitemap_url = url
        if not url.endswith(".xml"):
            parsed = urlparse(url)
            sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
        resp = await client.get(sitemap_url, headers={"User-Agent": settings.user_agent})
        if resp.status_code >= 400:
            return []
        urls = _parse_sitemap(resp.text)
        return [u for u in urls if _link_ok(u, source)][: source.max_pages]

    resp = await client.get(url, headers={"User-Agent": settings.user_agent})
    if resp.status_code >= 400:
        return []
    links = _extract_links(resp.text, url)
    filtered: list[str] = []
    for link in links:
        if not _same_origin(url, link):
            continue
        if not _within_path(url, link):
            continue
        if not _link_ok(link, source):
            continue
        filtered.append(link)
    unique = []
    seen = set()
    for link in filtered:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique[: source.max_pages]


async def scrape_project(
    project_id: str,
    sources: list[Source],
    allow_robots: bool,
    progress_cb: Optional[callable] = None,
) -> dict[str, Any]:
    limiter = RateLimiter(1.0 / max(settings.requests_per_host, 0.1))
    robots = RobotsCache()

    pages_path = project_dir(project_id) / "repo" / "pages.jsonl"
    if pages_path.exists():
        pages_path.unlink()
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    stats = {"ok": 0, "skipped": 0, "failed": 0, "words": 0, "discovered": 0}

    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        discovered: list[str] = []
        for source in sources:
            discovered.extend(await discover_urls(client, source))

        if not discovered:
            discovered = [normalize_url(str(source.url)) for source in sources]

        total = len(discovered)
        stats["discovered"] = total
        if progress_cb:
            progress_cb(0, total, "discovered")

        sem = asyncio.Semaphore(settings.max_concurrency)

        async def handle(url: str) -> None:
            nonlocal stats
            async with sem:
                if url in seen_urls:
                    stats["skipped"] += 1
                    return
                seen_urls.add(url)
                if allow_robots:
                    allowed = await robots.allowed(client, url)
                    if not allowed:
                        stats["skipped"] += 1
                        record = {
                            "id": _hash_url(url),
                            "url": url,
                            "source_root": urlparse(url).netloc,
                            "fetched_at": _now(),
                            "title": "",
                            "author": "",
                            "published_at": "",
                            "text": "",
                            "language": "",
                            "links_out": [],
                            "checksum": "",
                            "status": "skipped",
                            "error": "blocked_by_robots",
                        }
                        append_jsonl(pages_path, [record])
                        return

                try:
                    status, body, _ = await fetch_url(client, limiter, url, project_id)
                    if status >= 400 or status == 0:
                        stats["failed"] += 1
                        record = {
                            "id": _hash_url(url),
                            "url": url,
                            "source_root": urlparse(url).netloc,
                            "fetched_at": _now(),
                            "title": "",
                            "author": "",
                            "published_at": "",
                            "text": "",
                            "language": "",
                            "links_out": [],
                            "checksum": "",
                            "status": "error",
                            "error": f"http_{status}" if status else "request_error",
                        }
                        append_jsonl(pages_path, [record])
                        return

                    extracted = extract_content(body, url)
                    if not extracted.get("text"):
                        stats["failed"] += 1
                        record = {
                            "id": _hash_url(url),
                            "url": url,
                            "source_root": urlparse(url).netloc,
                            "fetched_at": _now(),
                            "title": extracted.get("title", ""),
                            "author": extracted.get("author", ""),
                            "published_at": extracted.get("published_at", ""),
                            "text": "",
                            "language": "",
                            "links_out": [],
                            "checksum": "",
                            "status": "error",
                            "error": "empty_text",
                        }
                        append_jsonl(pages_path, [record])
                        return

                    checksum = _hash_text(extracted["text"])
                    if checksum in seen_hashes:
                        stats["skipped"] += 1
                        return
                    seen_hashes.add(checksum)

                    record = {
                        "id": _hash_url(url),
                        "url": url,
                        "source_root": urlparse(url).netloc,
                        "fetched_at": _now(),
                        "title": extracted.get("title", ""),
                        "author": extracted.get("author", ""),
                        "published_at": extracted.get("published_at", ""),
                        "text": extracted.get("text", ""),
                        "language": "",
                        "links_out": [],
                        "checksum": checksum,
                        "status": "ok",
                        "error": "",
                    }
                    stats["ok"] += 1
                    stats["words"] += len(extracted.get("text", "").split())
                    append_jsonl(pages_path, [record])
                finally:
                    if progress_cb:
                        progress_cb(stats["ok"] + stats["skipped"] + stats["failed"], total, "scraping")

        await asyncio.gather(*(handle(url) for url in discovered))

    stats["updated_at"] = datetime.utcnow().isoformat()
    summary_path = project_dir(project_id) / "repo" / "scrape_summary.json"
    atomic_write_json(summary_path, stats)
    return stats
