"""Tests for the pipeline orchestration (all module stages are stubbed)."""

import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pipeline
from pipeline import PipelineError, PipelineResult, run_pipeline


# --- Fake stage outputs mimicking the real modules' result objects ----------

@dataclass
class FakeASRWord:
    word: str
    start: float
    end: float
    probability: float


class FakeTranscriptResult:
    def __init__(self):
        self.words = [FakeASRWord("hello", 0.0, 0.5, 0.95),
                      FakeASRWord("world", 0.5, 1.0, 0.9)]
        self.metadata = {"audio_duration_seconds": 1.0,
                         "processing_time_seconds": 2.0,
                         "word_count": 2, "segment_count": 1}

    def to_json_file(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{}", encoding="utf-8")
        return Path(path)


class FakeMatch:
    matched_text = "hello world"
    start_time = 0.0
    end_time = 1.0
    text_similarity = 1.0
    average_word_probability = 0.925
    minimum_word_probability = 0.9


class FakeFrame:
    frame_number = 25
    actual_timestamp = 1.0
    image_path = "frames/frame_25.jpg"
    fps = 25.0
    frame_count = 50
    duration = 2.0
    width = 320
    height = 240


class TestPipelineOrchestration(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_download(url):
            self.calls.append(("download", url))
            return Path("video/fake.mp4")

        def fake_audio(video_path):
            self.calls.append(("audio", video_path))
            assert Path(video_path) == Path("video") / "fake.mp4"
            return Path("audio/fake.wav")

        def fake_asr(wav_path, transcript_path, *a, **kw):
            self.calls.append(("asr", wav_path))
            assert Path(wav_path) == Path("audio") / "fake.wav"
            return FakeTranscriptResult()

        def fake_match(target, transcript_result):
            self.calls.append(("match", target))
            # Verify in-memory ASR words were passed (not reloaded from disk).
            assert isinstance(transcript_result, FakeTranscriptResult)
            return FakeMatch()

        def fake_frame(video_path, start_time):
            self.calls.append(("frame", video_path, start_time))
            assert start_time == 0.0
            return FakeFrame()

        patches = [
            mock.patch.object(pipeline, "run_download_stage", fake_download),
            mock.patch.object(pipeline, "run_audio_extraction_stage", fake_audio),
            mock.patch.object(pipeline, "run_asr_stage", fake_asr),
            mock.patch.object(pipeline, "run_matching_stage", fake_match),
            mock.patch.object(pipeline, "run_frame_extraction_stage", fake_frame),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_stage_order_and_data_flow(self):
        result = run_pipeline("http://example.com/v", "hello world",
                              save_result=False)
        order = [c[0] for c in self.calls]
        self.assertEqual(order, ["download", "audio", "asr", "match", "frame"])

    def test_result_fields(self):
        result = run_pipeline("http://example.com/v", "hello world",
                              save_result=False)
        self.assertIsInstance(result, PipelineResult)
        d = result.to_dict()
        for key in ("target_text", "matched_text", "timestamp", "frame_number",
                    "frame_image_path", "similarity", "average_word_probability",
                    "minimum_word_probability"):
            self.assertIn(key, d)
        self.assertEqual(result.matched_text, "hello world")
        self.assertEqual(result.timestamp, 0.0)
        self.assertEqual(result.frame_number, 25)
        self.assertEqual(result.frame_image_path, "frames/frame_25.jpg")
        self.assertAlmostEqual(result.similarity, 1.0)

    def test_stage_failure_stops_pipeline(self):
        def failing_frame(video_path, start_time):
            self.calls.append(("frame",))
            raise PipelineError("Frame extraction failed: boom")

        with mock.patch.object(pipeline, "run_frame_extraction_stage",
                               failing_frame):
            with self.assertRaises(PipelineError):
                run_pipeline("http://example.com/v", "hello world",
                             save_result=False)
        # Pipeline stopped at the last stage; nothing ran after it.
        self.assertEqual(self.calls[-1][0], "frame")

    def test_empty_url_rejected_before_any_stage(self):
        with self.assertRaises(PipelineError):
            run_pipeline("", "hello world", save_result=False)
        self.assertEqual(self.calls, [])

    def test_empty_target_rejected_before_any_stage(self):
        with self.assertRaises(PipelineError):
            run_pipeline("http://example.com/v", "   ", save_result=False)
        self.assertEqual(self.calls, [])

    def test_downloader_failure_wrapped(self):
        def failing_download(url):
            raise PipelineError("Video download failed: unreachable")

        with mock.patch.object(pipeline, "run_download_stage", failing_download):
            with self.assertRaises(PipelineError) as ctx:
                run_pipeline("http://bad", "hello world", save_result=False)
        self.assertIn("download failed", str(ctx.exception).lower())
        # No later stages executed.
        self.assertEqual(len(self.calls), 0)


class TestCLIValidation(unittest.TestCase):
    def test_missing_required_args_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            pipeline.main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_target_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            pipeline.main(["--url", "http://example.com/v"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
