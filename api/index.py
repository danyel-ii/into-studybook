import json
import os
import sys
from typing import Callable

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
API_DIR = os.path.join(BASE_DIR, "apps", "api")
if API_DIR not in sys.path:
    sys.path.append(API_DIR)


def _simple_error_app(detail: str) -> Callable:
    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


try:
    from fastapi import FastAPI, HTTPException
except Exception as exc:  # pragma: no cover
    detail = f"FastAPI import failed: {type(exc).__name__}: {exc}"
    print("API import failed", detail)
    app = _simple_error_app(detail)
else:
    try:
        from app.main import app as inner_app  # type: ignore
        app = FastAPI(title="SelfStudy API")
        # Mount under /api so /api/projects resolves when routed to this function.
        app.mount("/api", inner_app)
        # Optional: also mount at root for direct access.
        app.mount("/", inner_app)
    except Exception as exc:  # pragma: no cover
        # Fallback app to surface import errors in serverless logs.
        app = FastAPI(title="SelfStudy API (Import Error)")
        detail = f"{type(exc).__name__}: {exc}"
        print("API import failed", detail)

        @app.get("/api/health")
        async def health():
            raise HTTPException(status_code=500, detail=detail)

        @app.post("/api/projects")
        async def create_project():
            raise HTTPException(status_code=500, detail=detail)
