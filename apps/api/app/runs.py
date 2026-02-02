from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .storage import iso_now, project_dir


def save_run(
    project_id: str,
    run_type: str,
    prompt: str,
    response: Any,
    model: str,
    meta: Optional[dict] = None,
) -> Path:
    run_folder = project_dir(project_id) / "runs"
    run_folder.mkdir(parents=True, exist_ok=True)
    timestamp = iso_now().replace(":", "-")
    path = run_folder / f"{timestamp}-{run_type}.json"
    payload = {
        "run_type": run_type,
        "timestamp": iso_now(),
        "model": model,
        "prompt": prompt,
        "response": response,
        "meta": meta or {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path
