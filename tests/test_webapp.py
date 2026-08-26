"""Tests for the FastAPI web layer over the V2 pipeline (pipeline mocked)."""

import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import webapp.main as webmain
from webapp.jobs import JobStore


# ---------------------------------------------------------------------------
# JobStore unit tests
# ---------------------------------------------------------------------------

class TestJobStore(unittest.TestCase):
    def setUp(self):
        self.store = JobStore()

    def test_create_returns_queued_job_with_id(self):
        job_id = self.store.create()
        snap = self.store.snapshot(job_id)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"], "queued")
        self.assertIsNone(snap["result"])
        self.assertIsNone(snap["error"])
        self.assertEqual(snap["logs"], [])

    def test_lifecycle_queued_running_done(self):
        job_id = self.store.create()
        self.store.mark_running(job_id)
        self.assertEqual(self.store.snapshot(job_id)["state"], "running")

        self.store.set_stage(job_id, "coarse_asr", "Running coarse ASR...")
        snap = self.store.snapshot(job_id)
        self.assertEqual(snap["current_stage"], "coarse_asr")
        self.assertEqual(snap["message"], "Running coarse ASR...")
        self.assertEqual(snap["stages"]["coarse_asr"]["state"], "running")

        self.store.complete(job_id, result={"matched_text": "hi"})
        snap = self.store.snapshot(job_id)
        self.assertEqual(snap["state"], "done")
        self.assertEqual(snap["result"]["matched_text"], "hi")
        # A running stage left open must be closed on completion.
        self.assertEqual(snap["stages"]["coarse_asr"]["state"], "done")

    def test_failure_lifecycle_exposes_error(self):
        job_id = self.store.create()
        self.store.mark_running(job_id)
        self.store.fail(job_id, "Target dialogue not found: ...")
        snap = self.store.snapshot(job_id)
        self.assertEqual(snap["state"], "failed")
        self.assertIn("not found", snap["error"])
        self.assertIsNone(snap["result"])

    def test_progress_advances_stages_and_closes_previous(self):
        job_id = self.store.create()
        self.store.mark_running(job_id)
        self.store.apply_progress(job_id, "download", "Downloading video...")
        self.store.apply_progress(job_id, "audio_extraction", "Extracting audio...")
        snap = self.store.snapshot(job_id)
        self.assertEqual(snap["stages"]["download"]["state"], "done")
        self.assertEqual(snap["stages"]["audio_extraction"]["state"], "running")
        self.assertEqual(snap["current_stage"], "audio_extraction")

    def test_snapshot_is_a_copy(self):
        job_id = self.store.create()
        self.store.append_log(job_id, "line1")
        snap = self.store.snapshot(job_id)
        snap["logs"].append("mutated")
        stored = self.store.snapshot(job_id)["logs"]
        self.assertEqual(len(stored), 1)
        self.assertTrue(stored[0].endswith("line1"))

    def test_unknown_job_returns_none(self):
        self.assertIsNone(self.store.snapshot("nope"))

    def test_single_job_lock(self):
        self.assertTrue(self.store.try_begin())
        self.assertFalse(self.store.try_begin())  # second job rejected
        self.store.end_job()                      # releases
        self.assertTrue(self.store.try_begin())


# ---------------------------------------------------------------------------
# API tests (pipeline mocked - never runs the real thing)
# ---------------------------------------------------------------------------

def _fake_pipeline_result():
    from pipeline import PipelineResult

    return PipelineResult(
        target_text="hello world",
        matched_text="hello world",
        timestamp=325.29,
        frame_number=2745,
        frame_image_path=str(Path("frames") / "frame_2745.jpg"),
        similarity=1.0,
        average_word_probability=0.9,
        minimum_word_probability=0.68,
        video_path="video/v.mp4",
        audio_path="audio/a.wav",
        transcript_path="transcript/transcript.json",
        arch="v2",
        coarse_model="tiny",
        fine_model="small",
        fallback_tier="tiny->small",
        aligned=True,
        timestamp_hhmmss="00:05:25.290",
        stage_timings={"coarse_asr_s": 10.0, "fine_validation_s": 5.0},
    )


class TestApi(unittest.TestCase):
    def setUp(self):
        # Fresh app state per test.
        webmain.STORE = JobStore()
        self.client = TestClient(webmain.app)

    def test_post_validates_missing_url(self):
        resp = self.client.post("/api/localize", json={"url": "", "target": "x"})
        self.assertEqual(resp.status_code, 400)

    def test_post_validates_missing_target(self):
        resp = self.client.post("/api/localize", json={"url": "http://x", "target": "  "})
        self.assertEqual(resp.status_code, 400)

    @mock.patch.object(webmain, "run_pipeline_v2")
    def test_post_returns_job_id_immediately(self, mock_run):
        started = __import__("threading").Event()
        release = __import__("threading").Event()

        def slow_pipeline(*a, **kw):
            started.set()
            release.wait(timeout=5)
            return _fake_pipeline_result()

        mock_run.side_effect = slow_pipeline
        resp = self.client.post("/api/localize",
                                json={"url": "http://example.com/v",
                                      "target": "hello world"})
        self.assertEqual(resp.status_code, 200)
        job_id = resp.json()["job_id"]
        # POST must return before the worker finishes.
        self.assertIn(webmain.STORE.snapshot(job_id)["state"], ("queued", "running"))
        self.assertTrue(started.wait(timeout=5))
        release.set()

    @mock.patch.object(webmain, "run_pipeline_v2")
    def test_busy_server_returns_409(self, mock_run):
        # Simulate a long-running pipeline holding the lock inside the worker.
        started = __import__("threading").Event()
        release = __import__("threading").Event()

        def slow_pipeline(*a, **kw):
            started.set()
            release.wait(timeout=5)
            return _fake_pipeline_result()

        mock_run.side_effect = slow_pipeline
        first = self.client.post("/api/localize",
                                 json={"url": "http://x", "target": "t"})
        self.assertEqual(first.status_code, 200)
        self.assertTrue(started.wait(timeout=5))

        second = self.client.post("/api/localize",
                                  json={"url": "http://y", "target": "t"})
        self.assertEqual(second.status_code, 409)
        release.set()

    def test_unknown_job_404(self):
        resp = self.client.get("/api/jobs/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    @mock.patch.object(webmain, "run_pipeline_v2")
    def test_running_job_has_stage_and_message_fields(self, mock_run):
        hold = __import__("threading").Event()

        def running_pipeline(*a, **kw):
            progress = kw["progress"]
            progress("download", "Downloading video...")
            webmain.STORE.snapshot  # noqa: B018 (touch store existence)
            hold.wait(timeout=5)   # stay "running" while we assert
            return _fake_pipeline_result()

        mock_run.side_effect = running_pipeline
        job_id = self.client.post(
            "/api/localize", json={"url": "http://x", "target": "t"}).json()["job_id"]

        deadline = time.time() + 5
        data = None
        while time.time() < deadline:
            data = self.client.get(f"/api/jobs/{job_id}").json()
            if data["current_stage"] == "download":
                break
            time.sleep(0.05)
        hold.set()

        self.assertEqual(data["state"], "running")
        self.assertEqual(data["current_stage"], "download")
        self.assertIn("message", data)
        self.assertIn("logs", data)
        self.assertIsNone(data["result"])

    @mock.patch.object(webmain, "run_pipeline_v2")
    def test_completed_job_result_structure_and_image_url(self, mock_run):
        mock_run.return_value = _fake_pipeline_result()
        job_id = self.client.post(
            "/api/localize", json={"url": "http://x", "target": "t"}).json()["job_id"]

        deadline = time.time() + 5
        data = None
        while time.time() < deadline:
            data = self.client.get(f"/api/jobs/{job_id}").json()
            if data["state"] == "done":
                break
            time.sleep(0.05)

        self.assertEqual(data["state"], "done")
        result = data["result"]
        # Core fields reused from PipelineResult.to_dict().
        for key in ("matched_text", "timestamp", "frame_number", "similarity",
                    "frame_image_path", "timestamp_hhmmss", "fallback_tier",
                    "aligned"):
            self.assertIn(key, result)
        # Web-only field appended, not a competing schema.
        self.assertEqual(result["frame_image_url"], "/frames/frame_2745.jpg")
        self.assertIn("total_runtime_s", data)
        self.assertIn("stage_timings", result)

    @mock.patch.object(webmain, "run_pipeline_v2")
    def test_failed_job_exposes_error_not_stuck_running(self, mock_run):
        from pipeline import PipelineError

        mock_run.side_effect = PipelineError("Target dialogue not found: sim 0.2")
        job_id = self.client.post(
            "/api/localize", json={"url": "http://x", "target": "t"}).json()["job_id"]

        deadline = time.time() + 5
        data = None
        while time.time() < deadline:
            data = self.client.get(f"/api/jobs/{job_id}").json()
            if data["state"] == "failed":
                break
            time.sleep(0.05)

        self.assertEqual(data["state"], "failed")
        self.assertIn("not found", data["error"])
        self.assertIsNone(data["result"])
        # After failure the server accepts new jobs again.
        with mock.patch.object(webmain, "run_pipeline_v2",
                               return_value=_fake_pipeline_result()):
            ok = self.client.post("/api/localize",
                                  json={"url": "http://x", "target": "t"})
        self.assertEqual(ok.status_code, 200)

    @mock.patch.object(webmain, "run_pipeline_v2")
    def test_progress_callback_wired_into_pipeline(self, mock_run):
        hold = __import__("threading").Event()

        def fake_run(*a, **kw):
            progress = kw.get("progress")
            assert callable(progress)
            progress("download", "Downloading video...")
            progress("coarse_asr", "Running coarse ASR (tiny)...")
            hold.wait(timeout=5)  # stay mid-run while assertions execute
            return _fake_pipeline_result()

        mock_run.side_effect = fake_run
        job_id = self.client.post(
            "/api/localize", json={"url": "http://x", "target": "t"}).json()["job_id"]

        deadline = time.time() + 5
        snap = None
        while time.time() < deadline:
            snap = webmain.STORE.snapshot(job_id)
            if snap["current_stage"] == "coarse_asr":
                break
            time.sleep(0.05)
        hold.set()

        self.assertIsNotNone(snap)
        self.assertEqual(snap["stages"]["coarse_asr"]["state"], "running")
        # The earlier stage must have been closed by the progress hook.
        self.assertEqual(snap["stages"]["download"]["state"], "done")

    def test_frame_image_url_generation(self):
        self.assertEqual(
            webmain.frame_image_url(Path("frames") / "frame_2745.jpg"),
            "/frames/frame_2745.jpg")
        self.assertEqual(
            webmain.frame_image_url(Path("frames") / "my clip.jpg"),
            "/frames/my%20clip.jpg")


if __name__ == "__main__":
    unittest.main()
