"""Unit tests for the V1 frame extractor (synthetic + generated test video)."""

import json
import tempfile
import unittest
from pathlib import Path

from frame_extractor.core import (
    ExtractionError,
    FrameExtractor,
    FrameResult,
    MetadataError,
    VideoMetadata,
    ValidationError,
    frame_number_to_timestamp,
    get_video_metadata,
    timestamp_to_frame_number,
)


class TestTimestampToFrameNumber(unittest.TestCase):
    def test_zero_timestamp(self):
        self.assertEqual(timestamp_to_frame_number(0.0, 25.0), 0)

    def test_normal_timestamp(self):
        # 24.990s at 30 fps -> floor(749.7) = 749
        self.assertEqual(timestamp_to_frame_number(24.990, 30.0), 749)

    def test_exact_frame_boundary(self):
        self.assertEqual(timestamp_to_frame_number(2.0, 25.0), 50)

    def test_ntsc_fps(self):
        # 30000/1001 fps ≈ 29.97
        self.assertEqual(timestamp_to_frame_number(10.0, 30000 / 1001), 299)

    def test_inverse_roundtrip(self):
        ts = 12.3456
        fps = 29.97
        back = frame_number_to_timestamp(timestamp_to_frame_number(ts, fps), fps)
        self.assertLessEqual(back, ts)
        self.assertGreater(back, ts - 1 / fps)

    def test_invalid_fps_raises(self):
        with self.assertRaises(Exception):
            timestamp_to_frame_number(1.0, 0.0)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.extractor = FrameExtractor(output_dir=self.dir / "frames")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_video(self):
        with self.assertRaises(ValidationError):
            self.extractor.extract_at_timestamp(self.dir / "nope.mp4", 1.0)

    def test_empty_video_file(self):
        p = self.dir / "empty.mp4"
        p.write_bytes(b"")
        with self.assertRaises(ValidationError):
            self.extractor.extract_at_timestamp(p, 1.0)

    def test_negative_timestamp(self):
        with self.assertRaises(ValidationError):
            self.extractor.extract_at_timestamp("anything.mp4", -1.0)

    def test_non_numeric_timestamp(self):
        with self.assertRaises(ValidationError):
            self.extractor.extract_at_timestamp("anything.mp4", "abc")

    def test_corrupt_video_metadata(self):
        p = self.dir / "corrupt.mp4"
        p.write_bytes(b"\x00" * 1024)
        with self.assertRaises(MetadataError):
            self.extractor.extract_at_timestamp(p, 1.0)


class TestResultStructure(unittest.TestCase):
    def test_result_fields_and_types(self):
        r = FrameResult(
            timestamp=324.990, frame_number=8125, actual_timestamp=325.0,
            image_path="frames/frame_8125.jpg", fps=25.0, frame_count=10000,
            duration=400.0, width=1920, height=1080,
        )
        d = r.to_dict()
        for key in ("timestamp", "frame_number", "actual_timestamp",
                    "image_path", "fps", "frame_count", "duration",
                    "width", "height"):
            self.assertIn(key, d)
        self.assertIsInstance(d["frame_number"], int)


class TestIntegration(unittest.TestCase):
    """Integration: extract a real frame from a tiny generated video."""

    @classmethod
    def setUpClass(cls):
        from frame_extractor.core import find_tool
        import subprocess

        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.tmp.name) / "test_video.mp4"
        cmd = [
            find_tool("ffmpeg"), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
            "-pix_fmt", "yuv420p", str(cls.video),
        ]
        subprocess.run(cmd, capture_output=True, timeout=120, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_extract_real_frame(self):
        out_dir = Path(self.tmp.name) / "frames"
        extractor = FrameExtractor(output_dir=out_dir)
        result = extractor.extract_at_timestamp(self.video, 1.0)

        # Metadata from the actual generated video (25 fps, 2s, 320x240).
        self.assertAlmostEqual(result.fps, 25.0, places=3)
        self.assertAlmostEqual(result.duration, 2.0, places=1)
        self.assertEqual((result.width, result.height), (320, 240))
        self.assertEqual(result.frame_count, 50)

        # Frame selection: floor(1.0 * 25) = 25, actual = 25/25 = 1.0.
        self.assertEqual(result.frame_number, 25)
        self.assertAlmostEqual(result.actual_timestamp, 1.0)

        # Image was written and is non-trivial in size.
        image = Path(result.image_path)
        self.assertTrue(image.exists())
        self.assertGreater(image.stat().st_size, 0)
        self.assertEqual(image.parent, out_dir.resolve()
                         if out_dir.is_absolute() else out_dir)

    def test_timestamp_beyond_duration_rejected(self):
        extractor = FrameExtractor(output_dir=Path(self.tmp.name))
        with self.assertRaises(ValidationError):
            extractor.extract_at_timestamp(self.video, 5.0)

    def test_metadata_probe_matches_result(self):
        meta = get_video_metadata(self.video)
        self.assertIsInstance(meta, VideoMetadata)
        self.assertEqual(meta.width, 320)


if __name__ == "__main__":
    unittest.main()
