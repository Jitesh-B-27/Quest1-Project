"""FastAPI web layer over the V2 pipeline.

Run:
    uvicorn webapp.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000

Thin presentation layer only: POST /api/localize starts the existing
``run_pipeline_v2`` in a background worker thread; GET /api/jobs/{id} is
polled by the frontend for stage status and the final result.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline
from localizer.core import LocalizationError
from matcher.core import MatchNotFoundError
from pipeline import run_pipeline_v2
from webapp.jobs import JobStore

WEB_DIR = Path(__file__).resolve().parent / "static"
FRAMES_DIR = Path("frames").resolve()

STORE = JobStore()

app = FastAPI(title="Video Dialogue Locator")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame_image_url(image_path: str | Path) -> str:
    """Map a frame image path to its static URL under /frames."""
    from urllib.parse import quote

    return "/frames/" + quote(Path(image_path).name)


class LocalizeRequest(BaseModel):
    url: str
    target: str


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, url: str, target: str) -> None:
    store = STORE
    store.mark_running(job_id)

    def on_progress(stage: str, message: str) -> None:
        store.apply_progress(job_id, stage, message)

    def on_log(message: str) -> None:
        store.append_log(job_id, message)

    try:
        result = run_pipeline_v2(
            url=url,
            target=target,
            save_result=True,
            progress=on_progress,
            log=on_log,
        )
    except (pipeline.PipelineError, MatchNotFoundError,
            LocalizationError) as e:
        store.fail(job_id, str(e))
        return
    except Exception as e:  # never leave a job stuck in "running"
        store.fail(job_id, f"Unexpected pipeline error: {e}")
        return

    payload = result.to_dict()
    payload["frame_image_url"] = (
        frame_image_url(result.frame_image_path)
        if result.frame_image_path else None
    )
    store.complete(job_id, payload)

    for line in [f"Matched: {result.matched_text}",
                 f"Timestamp: {result.timestamp_hhmmss} "
                 f"({result.timestamp:.3f}s)",
                 f"Frame {result.frame_number}, similarity "
                 f"{result.similarity:.4f}, tier {result.fallback_tier}"]:
        store.append_log(job_id, line)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/localize")
def localize(req: LocalizeRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Missing video URL.")
    if not req.target or not req.target.strip():
        raise HTTPException(status_code=400, detail="Missing target dialogue.")

    if not STORE.try_begin():
        raise HTTPException(
            status_code=409,
            detail="A pipeline job is already running. Please wait for it "
                   "to finish.",
        )

    job_id = STORE.create()
    worker = threading.Thread(
        target=_worker_with_release, args=(job_id, req.url.strip(),
                                           req.target.strip()),
        daemon=True, name=f"job-{job_id}",
    )
    worker.start()
    return {"job_id": job_id}


def _worker_with_release(job_id: str, url: str, target: str) -> None:
    try:
        _run_job(job_id, url, target)
    finally:
        STORE.end_job()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    snap = STORE.snapshot(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="Unknown job ID.")
    if snap["state"] == "running" and snap["started_at"]:
        elapsed = time.time() - snap["started_at"]
    elif snap["state"] in ("done", "failed") and snap["finished_at"]:
        elapsed = snap["finished_at"] - (snap["started_at"] or snap["created_at"])
    else:
        elapsed = 0.0
    snap["elapsed_s"] = round(elapsed, 1)
    snap["total_runtime_s"] = snap["elapsed_s"]
    return snap
