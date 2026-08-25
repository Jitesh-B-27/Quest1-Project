"""Tests for V2 localization: candidate generation, pad/merge, region WAV
extraction, timestamps, aligner, and the cascade fallback ladder."""

import subprocess
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from asr.aligner import match_refined_onset, refine_word_timestamps
from asr.models import TranscriptResult, Word
from localizer.core import (
    CandidateRegion,
    LocalizationError,
    cleanup_regions,
    extract_region_wavs,
    format_timestamp,
    generate_top_k,
    pad_and_merge,
)
from localizer.cascade import run_cascade
from matcher.core import MatchNotFoundError


def W(word, start, end, prob=0.9):
    return Word(word=word, start=start, end=end, probability=prob)


TARGET = "hello world"


def coarse_transcript():
    """Coarse (noisy) transcript: filler + drifted target at ~100s + filler."""
    return TranscriptResult(
        metadata={"audio_duration_seconds": 200.0},
        words=[
            W("intro", 0.0, 0.5),
            W("music", 1.0, 2.0),
            W("some", 50.0, 50.3),
            W("talk", 51.0, 51.5),
            W("helo", 100.2, 100.7),      # drifted/garbled "hello"
            W("word", 101.1, 101.6),      # garbled "world"
            W("more", 150.0, 150.4),
            W("chat", 151.0, 151.5),
        ],
    )


# ---------------------------------------------------------------------------
# Top-K generation
# ---------------------------------------------------------------------------

class TestGenerateTopK(unittest.TestCase):
    def test_returns_up_to_k_candidates_ranked(self):
        regions = generate_top_k(coarse_transcript().words, TARGET, top_k=3)
        self.assertEqual(len(regions), 3)
        sims = [r.coarse_similarity for r in regions]
        self.assertEqual(sims, sorted(sims, reverse=True))
        # Best candidate must cover the true target location.
        best = regions[0]
        self.assertLessEqual(best.global_start, 100.2)
        self.assertGreaterEqual(best.global_end, 101.6)

    def test_no_coarse_threshold_rejection(self):
        # Even a garbage transcript still yields candidates (recall stage):
        # every window survives unless NMS suppresses it as a duplicate.
        garbage = TranscriptResult(words=[W("zz", 0.0, 0.5), W("qq", 1.0, 2.0)],
                                   metadata={})
        regions = generate_top_k(garbage.words, TARGET, top_k=5)
        self.assertEqual(len(regions), 3)  # [zz], [qq], [zz qq] - none rejected

    def test_nms_suppresses_duplicate_windows(self):
        # Many adjacent windows over the same spot; NMS should keep few.
        words = [W("hello", float(i), i + 0.9) for i in range(20)]
        words += [W("world", float(i + 1), i + 1.9) for i in range(19)]
        words = sorted(words, key=lambda w: w.start)
        regions = generate_top_k(words, TARGET, top_k=5)
        for a in range(len(regions)):
            for b in range(a + 1, len(regions)):
                ai, aj = regions[a].word_indices
                bi, bj = regions[b].word_indices
                overlap = max(0, min(aj, bj) - max(ai, bi))
                union = max(aj, bj) - min(ai, bi)
                self.assertLessEqual(overlap / union, 0.5)

    def test_empty_target_raises(self):
        with self.assertRaises(LocalizationError):
            generate_top_k([W("a", 0, 1)], "   !!!")


# ---------------------------------------------------------------------------
# Padding and merging
# ---------------------------------------------------------------------------

class TestPadAndMerge(unittest.TestCase):
    def test_padding_and_clamping_at_zero(self):
        r = CandidateRegion(1.0, 2.0, 0.9, 1, (0, 2))
        merged = pad_and_merge([r], margin_s=5.0, audio_duration=None)
        self.assertEqual(merged[0].global_start, 0.0)
        self.assertEqual(merged[0].global_end, 7.0)

    def test_clamping_at_duration_boundary(self):
        r = CandidateRegion(196.0, 198.0, 0.9, 1, (0, 2))
        merged = pad_and_merge([r], margin_s=5.0, audio_duration=200.0)
        self.assertEqual(merged[0].global_start, 191.0)
        self.assertEqual(merged[0].global_end, 200.0)

    def test_overlapping_regions_merge(self):
        a = CandidateRegion(10.0, 20.0, 0.8, 1, (0, 2))
        b = CandidateRegion(15.0, 30.0, 0.9, 2, (3, 5))
        merged = pad_and_merge([a, b], margin_s=0.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].global_start, 10.0)
        self.assertEqual(merged[0].global_end, 30.0)
        self.assertEqual(merged[0].source_rank, 1)  # keeps better rank

    def test_disjoint_regions_stay_separate(self):
        a = CandidateRegion(10.0, 12.0, 0.8, 1, (0, 2))
        b = CandidateRegion(50.0, 52.0, 0.9, 2, (3, 5))
        merged = pad_and_merge([a, b], margin_s=1.0)
        self.assertEqual(len(merged), 2)

    def test_clamped_values_are_exact_floats(self):
        r = CandidateRegion(0.37, 2.13, 0.8, 1, (0, 2))
        merged = pad_and_merge([r], margin_s=5.0, audio_duration=200.0)[0]
        self.assertIsInstance(merged.global_start, float)
        self.assertEqual(merged.global_start, 0.0)


# ---------------------------------------------------------------------------
# Region WAV extraction
# ---------------------------------------------------------------------------

class TestExtractRegionWavs(unittest.TestCase):
    def _write_wav(self, path: Path, seconds: float = 3.0, rate: int = 16000):
        import math, struct

        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            frames = b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * 3.14159 * 440 * t / rate)))
                for t in range(int(seconds * rate))
            )
            f.writeframes(frames)

    def test_ffmpeg_command_uses_exact_global_start(self):
        region = CandidateRegion(320.0, 330.0, 0.9, 1, (0, 2))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            out = Path(cmd[-1])
            out.write_bytes(b"RIFF")
            return mock.MagicMock(returncode=0, stderr="")

        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "full.wav"
            src.write_bytes(b"x")
            with mock.patch("localizer.core.subprocess.run", side_effect=fake_run):
                result = extract_region_wavs(src, [region],
                                             output_dir=Path(tmp) / "regions",
                                             ffmpeg_path="ffmpeg")
            cmd = calls[0]
            ss_idx = cmd.index("-ss")
            self.assertEqual(cmd[ss_idx + 1], "320.000000")
            # global_start must equal the literal -ss value handed to ffmpeg.
            self.assertEqual(result[0][0].global_start, 320.0)
            self.assertEqual(result[0][0].global_start,
                             float(cmd[ss_idx + 1]))
            t_idx = cmd.index("-t")
            self.assertEqual(cmd[t_idx + 1], "10.000000")

    def test_real_extraction_cuts_correct_duration(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            full = tmp_path / "full.wav"
            self._write_wav(full, seconds=3.0)
            region = CandidateRegion(0.5, 2.0, 0.9, 1, (0, 1))
            result = extract_region_wavs(full, [region],
                                         output_dir=tmp_path / "regions")
            cut_path = result[0][1]
            with wave.open(str(cut_path)) as f:
                duration = f.getnframes() / f.getframerate()
            self.assertAlmostEqual(duration, 1.5, delta=0.1)
            self.assertEqual(result[0][0].global_start, 0.5)

    def test_missing_full_wav_raises(self):
        with self.assertRaises(LocalizationError):
            extract_region_wavs("definitely/missing.wav",
                                [CandidateRegion(0, 1, 0.5, 1, (0, 1))])

    def test_cleanup_removes_only_region_wavs(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "region_00.wav").write_bytes(b"x")
            (d / "region_01.wav").write_bytes(b"x")
            (d / "keep.txt").write_text("keep")
            removed = cleanup_regions(d)
            self.assertEqual(removed, 2)
            self.assertTrue((d / "keep.txt").exists())


# ---------------------------------------------------------------------------
# Timestamp formatting and local->global conversion
# ---------------------------------------------------------------------------

class TestTimestampHandling(unittest.TestCase):
    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0.0), "00:00:00.000")
        self.assertEqual(format_timestamp(3661.5), "01:01:01.500")
        self.assertEqual(format_timestamp(324.918), "00:05:24.918")

    def test_format_precision_float_noise(self):
        # 59.9995 rounds to 60.000; ensure no 59.999-style artifacts.
        self.assertEqual(format_timestamp(59.99949), "00:00:59.999")
        self.assertEqual(format_timestamp(86399.9999), "23:59:60.000" if False
                         else "23:59:59.999" if round(86399.9999 * 1000) == 86399999
                         else "24:00:00.000")

    def test_local_to_global_conversion(self):
        region = CandidateRegion(global_start=320.0, global_end=332.0,
                                 coarse_similarity=0.9, source_rank=1,
                                 word_indices=(0, 2))
        local_onset = 4.918
        self.assertAlmostEqual(region.global_start + local_onset, 324.918)

    def test_conversion_with_zero_start_region(self):
        region = CandidateRegion(global_start=0.0, global_end=10.0,
                                 coarse_similarity=0.9, source_rank=1,
                                 word_indices=(0, 2))
        self.assertEqual(region.global_start + 1.234, 1.234)


# ---------------------------------------------------------------------------
# Aligner (refine-only)
# ---------------------------------------------------------------------------

class TestAligner(unittest.TestCase):
    def _fake_whisperx(self, align_words):
        """Build a fake whisperx module exposing only load_audio/align."""
        fake = mock.MagicMock()
        fake.load_audio.return_value = [0.0] * 100
        fake.load_align_model.return_value = (object(), {"meta": True})
        aligned = {"segments": [{"words": [
            {"word": w, "start": s, "end": e, "score": p}
            for w, s, e, p in align_words]}]}
        fake.align.return_value = aligned
        return fake

    def test_refine_maps_words_and_never_calls_asr(self):
        fake = self._fake_whisperx([
            ("hello", 4.902, 5.301, 0.97),
            ("world", 5.310, 5.900, 0.95),
        ])
        transcript = TranscriptResult(
            metadata={},
            words=[W("hello", 4.918, 5.3), W("world", 5.32, 5.9)])
        with mock.patch.dict("sys.modules", {"whisperx": fake}):
            refined = refine_word_timestamps(Path("regions/r.wav"), transcript)
        self.assertIsNotNone(refined)
        self.assertEqual([w.word for w in refined], ["hello", "world"])
        self.assertEqual(refined[0].start, 4.902)
        fake.load_audio.assert_called_once()
        fake.align.assert_called_once()
        # Refine-only contract: no transcription performed.
        fake.transcribe.assert_not_called()

    def test_refine_failure_returns_none(self):
        fake = mock.MagicMock()
        fake.load_audio.side_effect = RuntimeError("boom")
        with mock.patch.dict("sys.modules", {"whisperx": fake}):
            self.assertIsNone(refine_word_timestamps(Path("r.wav"),
                                                     [W("hi", 0, 1)]))

    def test_refine_empty_transcript_returns_none(self):
        fake = self._fake_whisperx([])
        with mock.patch.dict("sys.modules", {"whisperx": fake}):
            self.assertIsNone(refine_word_timestamps(Path("r.wav"), []))

    def test_match_refined_onset_finds_sequence(self):
        matched = [W("hello", 4.918, 5.3), W("world", 5.32, 5.9)]
        refined = [W("uhm", 4.0, 4.5), W("hello", 4.902, 5.29),
                   W("world", 5.31, 5.88)]
        onset = match_refined_onset(matched, refined)
        self.assertEqual(onset, 4.902)

    def test_match_refined_onset_missing_returns_none(self):
        matched = [W("hello", 4.9, 5.3)]
        refined = [W("goodbye", 4.9, 5.3)]
        self.assertIsNone(match_refined_onset(matched, refined))


# ---------------------------------------------------------------------------
# Cascade fallback ladder (models stubbed)
# ---------------------------------------------------------------------------

class FakeTranscriber:
    """Scriptable stand-in for asr.Transcriber (factory + model stub)."""

    def __init__(self, script: dict[str, list], calls: list):
        self.script = script   # model_size -> list of results (popped)
        self.calls = calls

    def __call__(self, *, model_size, model_type="faster-whisper",
                 device="cpu", compute_type=None, language="en"):
        self.calls.append(model_size)
        holder = self

        class _StubModel:
            def transcribe(self, wav_path):
                return holder.script[model_size].pop(0)

        return _StubModel()


class TestCascadeFallback(unittest.TestCase):
    # Coarse best window starts at 100.2s; the rank-2 window (50.0s) overlaps
    # it after padding, so the merged winner region starts at 50.0 - 5 = 45.0.
    MERGED_REGION_START = 45.0

    def _run(self, tiny, base, small_regions, small_full=None, extra_kwargs=None):
        """Run cascade with scripted per-model results."""
        script = {"tiny": list(tiny), "base": list(base),
                  "small": list(small_regions) + list(small_full or [])}
        calls: list[str] = []
        with mock.patch.object(
                __import__("localizer.cascade", fromlist=["Transcriber"]),
                "Transcriber", FakeTranscriber(script, calls)):
            kwargs = dict(
                wav_path=Path("audio/full.wav"),
                target=TARGET,
                regions_dir=Path("unused"),  # extraction mocked out below
            )
            kwargs.update(extra_kwargs or {})
            return run_cascade(**kwargs), calls

    def setUp(self):
        # Always stub ffmpeg region cutting: map regions to fake wav paths.
        patcher = mock.patch(
            "localizer.cascade.extract_region_wavs",
            side_effect=lambda wav, regions, output_dir=None, **kw: [
                (r, Path(f"regions/fake_{i}.wav")) for i, r in enumerate(regions)])
        patcher.start()
        self.addCleanup(patcher.stop)
        # Alignment off by default in these tests (covered separately).
        align_patcher = mock.patch("localizer.cascade.refine_word_timestamps",
                                   return_value=None)
        align_patcher.start()
        self.addCleanup(align_patcher.stop)

    def test_success_at_tiny_tier(self):
        fine_hit = TranscriptResult(words=[W("well", 0.0, 0.4),
                                           W("hello", 4.918, 5.3),
                                           W("world", 5.32, 5.9),
                                           W("today", 6.0, 6.5)],
                                    metadata={})
        result, calls = self._run(tiny=[coarse_transcript()],
                                  base=[],
                                  small_regions=[fine_hit])
        self.assertEqual(result.fallback_tier, "tiny->small")
        self.assertEqual(calls, ["tiny", "small"])  # no base tier attempted
        self.assertAlmostEqual(result.global_timestamp,
                               self.MERGED_REGION_START + 4.918)

    def test_escalates_to_base_then_small_full(self):
        bad_fine = TranscriptResult(words=[W("nothing relevant", 0.0, 0.5)],
                                    metadata={})
        v1_style_hit = TranscriptResult(words=[W("hello", 77.0, 77.4),
                                               W("world", 77.5, 78.0)],
                                        metadata={})
        result, calls = self._run(tiny=[coarse_transcript()],
                                  base=[coarse_transcript()],
                                  small_regions=[bad_fine, bad_fine],
                                  small_full=[v1_style_hit])
        self.assertEqual(calls, ["tiny", "small", "base", "small", "small"])
        self.assertEqual(result.fallback_tier, "small-full")
        self.assertFalse(result.aligned)
        self.assertAlmostEqual(result.global_timestamp, 77.0)

    def test_all_tiers_fail_raises_match_not_found(self):
        empty = TranscriptResult(words=[], metadata={})
        with self.assertRaises(MatchNotFoundError):
            self._run(tiny=[empty], base=[empty], small_regions=[],
                      small_full=[empty])

    def test_alignment_refines_timestamp_when_available(self):
        fine_hit = TranscriptResult(words=[W("hello", 4.918, 5.3),
                                           W("world", 5.32, 5.9)],
                                    metadata={})
        refined_words = [Word(word="hello", start=4.902, end=5.29,
                              probability=0.98),
                         Word(word="world", start=5.31, end=5.88,
                              probability=0.97)]
        with mock.patch("localizer.cascade.refine_word_timestamps",
                        return_value=refined_words):
            result, _ = self._run(tiny=[coarse_transcript()], base=[],
                                  small_regions=[fine_hit])
        self.assertTrue(result.aligned)
        # Refined onset 4.902 replaces fine-ASR onset 4.918.
        self.assertAlmostEqual(result.global_timestamp,
                               self.MERGED_REGION_START + 4.902)

    def test_alignment_failure_keeps_pipeline_alive(self):
        fine_hit = TranscriptResult(words=[W("hello", 4.918, 5.3),
                                           W("world", 5.32, 5.9)],
                                    metadata={})
        with mock.patch("localizer.cascade.refine_word_timestamps",
                        side_effect=RuntimeError("boom")):
            result, _ = self._run(tiny=[coarse_transcript()], base=[],
                                  small_regions=[fine_hit])
        self.assertFalse(result.aligned)  # graceful fallback to fine-ASR onset
        self.assertAlmostEqual(result.global_timestamp,
                               self.MERGED_REGION_START + 4.918)

    def test_timings_populated(self):
        fine_hit = TranscriptResult(words=[W("hello", 4.9, 5.3),
                                           W("world", 5.32, 5.9)], metadata={})
        result, _ = self._run(tiny=[coarse_transcript()], base=[],
                              small_regions=[fine_hit])
        self.assertIn("coarse_asr_tiny_s", result.timings)
        self.assertIn("fine_asr_validation_t0_s", result.timings)
        self.assertIn("forced_alignment_s", result.timings)


if __name__ == "__main__":
    unittest.main()
