"""Minimal in-memory job store for the web layer (single worker thread).

Thread-safety: every read goes through ``snapshot`` which copies mutable
fields under the store lock; writers mutate under the same lock.
"""

from __future__ import annotations

import threading
import time
import uuid

# Canonical stage order used by the UI timeline.
STAGE_ORDER = [
    "download",
    "audio_extraction",
    "coarse_asr",
    "candidate_generation",
    "fine_validation",
    "alignment",
    "frame_extraction",
]


class JobStore:
    """In-memory job registry + single-pipeline lock."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()  # one pipeline at a time

    # -- lifecycle ---------------------------------------------------------

    def create(self) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "state": "queued",
                "current_stage": None,
                "message": "Queued...",
                "stages": {key: {"state": "pending", "message": ""}
                           for key in STAGE_ORDER},
                "logs": [],
                "result": None,
                "error": None,
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
            }
        return job_id

    def try_begin(self) -> bool:
        """Claim the single pipeline slot. False if a job is already running."""
        return self._worker_lock.acquire(blocking=False)

    def end_job(self) -> None:
        if self._worker_lock.locked():
            self._worker_lock.release()

    # -- mutation ----------------------------------------------------------

    def _job(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._job(job_id)
            if job:
                job["state"] = "running"
                job["started_at"] = time.time()

    def set_stage(self, job_id: str, stage: str, message: str,
                  state: str = "running") -> None:
        with self._lock:
            job = self._job(job_id)
            if not job:
                return
            job["current_stage"] = stage
            job["message"] = message
            if stage in job["stages"]:
                job["stages"][stage]["state"] = state
                job["stages"][stage]["message"] = message

    def apply_progress(self, job_id: str, stage: str, message: str) -> None:
        """Advance the stage timeline: close the previous running stage."""
        with self._lock:
            job = self._job(job_id)
            if not job or job["state"] != "running":
                return
            for key, info in job["stages"].items():
                if info["state"] == "running" and key != stage:
                    info["state"] = "done"
            job["current_stage"] = stage
            job["message"] = message
            if stage in job["stages"]:
                job["stages"][stage]["state"] = "running"
                job["stages"][stage]["message"] = message

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._job(job_id)
            if job:
                job["logs"].append(f"[{time.strftime('%H:%M:%S')}] {line}")
                del job["logs"][:-200]  # cap memory

    def complete(self, job_id: str, result: dict) -> None:
        with self._lock:
            job = self._job(job_id)
            if not job:
                return
            job["state"] = "done"
            job["result"] = result
            job["message"] = "Complete"
            job["finished_at"] = time.time()
            for info in job["stages"].values():
                if info["state"] == "running":
                    info["state"] = "done"

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._job(job_id)
            if not job:
                return
            job["state"] = "failed"
            job["error"] = error
            job["message"] = error
            job["finished_at"] = time.time()

    # -- access ------------------------------------------------------------

    def snapshot(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            snap = dict(job)
            snap["logs"] = list(job["logs"])
            snap["stages"] = {k: dict(v) for k, v in job["stages"].items()}
            snap["result"] = dict(job["result"]) if job["result"] else None
            return snap
