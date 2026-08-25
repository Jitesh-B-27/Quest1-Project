"""Tests for pipeline media caching (skip download / audio stages)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pipeline
from pipeline import PipelineError, resolve_media_path, run_pipeline


class FakeTranscriptResult:
    words = []

    def to_json_file(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("{}", encoding="utf-8")
        return Path(path)


def _fake_match():
    m = mock.MagicMock()
    m.matched_text = "hello world"
    m.start_time = 0.0
    m.end_time = 1.0
    m.text_similarity = 1.0
    m.average_word_probability = 0.9
    m.minimum_word_probability = 0.9
    return m


def _fake_frame(video_path):
    f = mock.MagicMock()
    f.frame_number = 25
    f.actual_timestamp = 0.0
    f.image_path = Path("frames/frame_25.jpg")
    f.fps = 25.0
    f.frame_count = 50
    f.duration = 2.0
    f.width = 320
    f.height = 240
    return f


class TestResolveMediaPath(unittest.TestCase):
    def test_file_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "clip.mp4"
            f.write_bytes(b"x")
            self.assertEqual(resolve_media_path(f, {".mp4"}, "video"), f)

    def test_directory_picks_first_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "b.wav").write_bytes(b"x")
            (Path(tmp) / "a.wav").write_bytes(b"x")
            got = resolve_media_path(tmp, {".wav"}, "audio")
            self.assertEqual(got.name, "a.wav")

    def test_missing_path_raises(self):
        with self.assertRaises(PipelineError):
            resolve_media_path("definitely/not/here", {".mp4"}, "video")

    def test_directory_without_match_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text("hi")
            with self.assertRaises(PipelineError):
                resolve_media_path(tmp, {".mp4"}, "video")


class TestPipelineCacheReuse(unittest.TestCase):
    def _patch_stages(self):
        calls = []

        def fake_download(url, proxy=None):
            calls.append(("download", url))
            return Path("video/fake.mp4")

        def fake_audio(video_path):
            calls.append(("audio", video_path))
            return Path("audio/fake.wav")

        fake_asr = lambda wav, tp, *a, **kw: (
            calls.append(("asr", wav)), FakeTranscriptResult())[1]
        fake_match = lambda target, tr: (calls.append(("match", target)),
                                         _fake_match())[1]
        fake_frame = lambda vp, st: (calls.append(("frame", vp)),
                                     _fake_frame(vp))[1]

        for name, obj in [("run_download_stage", fake_download),
                          ("run_audio_extraction_stage", fake_audio),
                          ("run_asr_stage", fake_asr),
                          ("run_matching_stage", fake_match),
                          ("run_frame_extraction_stage", fake_frame)]:
            p = mock.patch.object(pipeline, name, obj)
            p.start()
            self.addCleanup(p.stop)
        return calls

    def test_cached_video_and_audio_skip_stages_1_and_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "ep.mp4"
            audio = Path(tmp) / "ep.wav"
            video.write_bytes(b"v")
            audio.write_bytes(b"a")
            calls = self._patch_stages()
            run_pipeline(target="hello world", save_result=False,
                         video_path=str(video), audio_path=str(audio))
            kinds = [c[0] for c in calls]
            self.assertNotIn("download", kinds)
            self.assertNotIn("audio", kinds)
            self.assertEqual(kinds[:2], ["asr", "match"])

    def test_cached_video_skips_only_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "ep.mkv"
            video.write_bytes(b"v")
            calls = self._patch_stages()
            run_pipeline(url="http://unused", target="hello world",
                         save_result=False, video_path=str(video))
            kinds = [c[0] for c in calls]
            self.assertNotIn("download", kinds)
            self.assertEqual(kinds[0], "audio")

    def test_no_url_and_no_video_raises(self):
        self._patch_stages()
        with self.assertRaises(PipelineError):
            run_pipeline(url=None, target="hello world", save_result=False)


if __name__ == "__main__":
    unittest.main()
