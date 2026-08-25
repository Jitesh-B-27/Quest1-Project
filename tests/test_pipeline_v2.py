"""Integration tests for the V2 pipeline orchestration."""

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pipeline
import localizer.cascade as cascade_module
from asr.models import TranscriptResult
from matcher.core import Word
from pipeline import PipelineError, run_pipeline_v2
from test_localizer import coarse_transcript


@dataclass
class FakeFrame:
    frame_number = 2745
    actual_timestamp = 104.916
    image_path = "frames/frame_2745.jpg"
    fps = 25.0
    frame_count = 50000
    duration = 2000.0
    width = 640
    height = 480


class TestPipelineV2(unittest.TestCase):
    def test_v2_passes_global_timestamp_to_frame_extractor(self):
        """The winning region's GLOBAL timestamp (not the local one) must
        reach FrameExtractor, with millisecond precision."""
        fine_hit = TranscriptResult(
            metadata={},
            words=[Word("hello", 4.918, 5.3, 0.95),
                   Word("world", 5.32, 5.9, 0.9)],
        )

        real_generate = cascade_module.generate_top_k
        real_pad = cascade_module.pad_and_merge

        def fake_extract(wav, regions, output_dir=None, **kw):
            # Simulate ffmpeg cutting: region[0].global_start stays exact.
            return [(r, Path(f"regions/fake_{i}.wav"))
                    for i, r in enumerate(regions)]

        captured = {}

        def fake_frame(video_path, start_time):
            captured["video"] = str(video_path)
            captured["timestamp"] = start_time
            return FakeFrame()

        with mock.patch.object(pipeline, "run_download_stage"), \
             mock.patch.object(pipeline, "run_audio_extraction_stage",
                               return_value=Path("audio/full.wav")), \
             mock.patch.object(cascade_module, "Transcriber") as fake_tr, \
             mock.patch.object(cascade_module, "extract_region_wavs",
                               side_effect=fake_extract), \
             mock.patch.object(cascade_module, "refine_word_timestamps",
                               return_value=None), \
             mock.patch.object(pipeline, "run_frame_extraction_stage",
                               side_effect=fake_frame):
            # Coarse call -> tiny transcript; fine call -> region transcript.
            fake_tr.return_value.transcribe.side_effect = [
                coarse_transcript(), fine_hit]

            with TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                video = tmp_path / "cached.mp4"
                audio = tmp_path / "full.wav"
                video.write_bytes(b"v")   # cache resolution only checks existence
                audio.write_bytes(b"a")
                result = run_pipeline_v2(
                    url=None,
                    target="hello world",
                    save_result=False,
                    video_path=video,
                    audio_path=audio,
                    regions_dir=tmp_path / "regions",
                )

        # Merged winner region starts at (50.0 - 5.0) = 45.0s; the fine-ASR
        # local onset of "hello" is 4.918s.
        expected_global = 45.0 + 4.918
        self.assertEqual(captured["timestamp"], result.timestamp)
        self.assertAlmostEqual(result.timestamp, expected_global, places=3)
        self.assertEqual(result.arch, "v2")
        self.assertEqual(result.fallback_tier, "tiny->small")
        self.assertFalse(result.aligned)
        self.assertEqual(len(result.timestamp_hhmmss.split(":")), 3)
        self.assertTrue(result.timestamp_hhmmss.endswith(".918"))

    def test_v1_still_default_and_working(self):
        calls = []

        def fake_download(url, proxy=None):
            return Path("video/fake.mp4")

        def fake_audio(video_path):
            return Path("audio/fake.wav")

        class FakeResult:
            words = []
            metadata = {}

            def to_json_file(self, p):
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                Path(p).write_text("{}", encoding="utf-8")
                return Path(p)

        @dataclass
        class FakeMatch:
            matched_text: str = "hello world"
            start_time: float = 10.0
            end_time: float = 11.0
            text_similarity: float = 1.0
            average_word_probability: float = 0.9
            minimum_word_probability: float = 0.9

        patches = [
            mock.patch.object(pipeline, "run_download_stage", fake_download),
            mock.patch.object(pipeline, "run_audio_extraction_stage", fake_audio),
            mock.patch.object(pipeline, "run_asr_stage",
                              lambda *a, **k: FakeResult()),
            mock.patch.object(pipeline, "run_matching_stage",
                              lambda *a, **k: FakeMatch()),
            mock.patch.object(pipeline, "run_frame_extraction_stage",
                              side_effect=lambda vp, st:
                              (calls.append(("frame", vp, st)), FakeFrame())[1]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        result = pipeline.run_pipeline(url="http://example.com/v",
                                       target="hello world", save_result=False)
        self.assertEqual(result.arch, "v1")
        self.assertEqual(calls[-1][2], 10.0)  # V1 timestamp path unchanged

    def test_cli_arch_flag_dispatch(self):
        with mock.patch.object(pipeline, "run_pipeline_v2") as v2, \
             mock.patch.object(pipeline, "run_pipeline") as v1:
            pipeline.main(["--arch", "v2", "--target", "hi"])
            v2.assert_called_once()
            v1.assert_not_called()
        with mock.patch.object(pipeline, "run_pipeline_v2") as v2, \
             mock.patch.object(pipeline, "run_pipeline") as v1:
            pipeline.main(["--arch", "v1", "--target", "hi",
                           "--url", "http://x"])
            v1.assert_called_once()
            v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
