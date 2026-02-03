import os
import sys
from fastapi import FastAPI, HTTPException

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
API_DIR = os.path.join(BASE_DIR, "apps", "api")
if API_DIR not in sys.path:
    sys.path.append(API_DIR)

try:
    from app.main import app  # type: ignore
except Exception as exc:  # pragma: no cover
    # Fallback app to surface import errors in serverless logs.
    app = FastAPI(title="SelfStudy API (Import Error)")
    detail = f"{type(exc).__name__}: {exc}"
    print("API import failed", detail)

    @app.get("/health")
    async def health():
        raise HTTPException(status_code=500, detail=detail)

    @app.post("/projects")
    async def create_project():
        raise HTTPException(status_code=500, detail=detail)
