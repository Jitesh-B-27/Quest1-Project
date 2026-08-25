"""Tests for the pluggable ASR providers (heavy inference is mocked)."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import wave

from asr.exceptions import DependencyError, ValidationError
from asr.factory import PROVIDERS, create_asr_provider
from asr.models import TranscriptResult
from asr.providers.faster_whisper_provider import FasterWhisperProvider
from asr.providers.parakeet_provider import ParakeetProvider
from asr.providers.whisperx_provider import WhisperXProvider


def make_tiny_wav(directory: Path) -> Path:
    """Create a minimal valid WAV file for validation checks."""
    p = directory / "test.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1 s of silence
    return p


class TestFactory(unittest.TestCase):
    def test_selects_correct_provider(self):
        self.assertIsInstance(
            create_asr_provider("faster-whisper"), FasterWhisperProvider)
        self.assertIsInstance(
            create_asr_provider("whisperx"), WhisperXProvider)
        self.assertIsInstance(
            create_asr_provider("parakeet"), ParakeetProvider)

    def test_case_insensitive(self):
        self.assertIsInstance(
            create_asr_provider("Faster-Whisper"), FasterWhisperProvider)

    def test_unsupported_provider_rejected(self):
        with self.assertRaises(ValidationError):
            create_asr_provider("sky-net")

    def test_unsupported_model_rejected(self):
        with self.assertRaises(ValidationError):
            create_asr_provider("faster-whisper", model="gpt-4")

    def test_default_models(self):
        self.assertEqual(create_asr_provider("faster-whisper").model, "small")
        self.assertEqual(create_asr_provider("whisperx").model, "small")
        p = create_asr_provider("parakeet")
        self.assertTrue(p.model.startswith("nvidia/parakeet"))

    def test_parakeet_rejects_whisper_size_keywords(self):
        with self.assertRaises(ValidationError) as ctx:
            create_asr_provider("parakeet", model="small")
        msg = str(ctx.exception)
        self.assertIn("HuggingFace", msg)
        self.assertIn("nvidia/parakeet-tdt-0.6b-v3", msg)

    def test_parakeet_accepts_repo_names(self):
        for name in ("nvidia/parakeet-tdt-0.6b-v3",
                     "nvidia/parakeet-tdt-0.6b-v2"):
            provider = create_asr_provider("parakeet", model=name)
            self.assertEqual(provider.model, name)

    def test_registry_has_all_providers(self):
        self.assertEqual(
            set(PROVIDERS),
            {"faster-whisper", "whisperx", "parakeet"},
        )


class TestWhisperXProvider(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = make_tiny_wav(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _install_fake_whisperx(self):
        """Inject a fake whisperx module producing deterministic aligned words."""
        fake_module = mock.MagicMock()
        fake_module.load_audio.return_value = b"fake-audio"
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"text": "Hello world.",
                 "words": [
                     {"word": "Hello", "start": 0.0, "end": 0.4,
                      "probability": 0.91},
                     {"word": "world.", "start": 0.5, "end": 0.9,
                      "probability": 0.88},
                 ]},
            ],
        }
        fake_module.load_model.return_value = fake_model
        align_model, align_meta = mock.MagicMock(), mock.MagicMock()
        fake_module.load_align_model.return_value = (align_model, align_meta)

        def fake_align(segments, am, ameta, audio, device,
                       return_char_alignments=False):
            return {
                "segments": [
                    {"words": [
                        {"word": "Hello", "start": 0.01, "end": 0.41,
                         "score": 0.97},
                        {"word": "world.", "start": 0.52, "end": 0.93,
                         "score": 0.95},
                    ]},
                ],
            }

        fake_module.align.side_effect = fake_align
        return {"whisperx": fake_module}

    def test_returns_transcript_result_with_aligned_words(self):
        provider = WhisperXProvider(model="small")
        with mock.patch.dict(sys.modules, self._install_fake_whisperx()):
            result = provider.transcribe(self.wav)

        self.assertIsInstance(result, TranscriptResult)
        self.assertEqual(len(result.words), 2)
        d = result.words[0].to_dict()
        self.assertEqual(set(d), {"word", "start", "end", "probability"})
        # Alignment score is preferred over raw transcription probability.
        self.assertAlmostEqual(result.words[0].probability, 0.97)
        self.assertAlmostEqual(result.words[1].probability, 0.95)
        self.assertEqual(result.metadata["provider"], "whisperx")
        self.assertEqual(result.metadata["word_count"], 2)
        self.assertEqual(result.metadata["detected_language"], "en")

    def test_dependency_error_when_not_installed(self):
        with mock.patch.dict(sys.modules, {"whisperx": None}):
            with self.assertRaises(DependencyError):
                WhisperXProvider().transcribe(self.wav)


class TestParakeetProvider(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = make_tiny_wav(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _install_fake_nemo(self, hypothesis):
        fake_asr_models = mock.MagicMock()
        fake_nemo_model = mock.MagicMock()

        def fake_transcribe(paths, **kwargs):
            return [hypothesis]

        fake_nemo_model.transcribe.side_effect = fake_transcribe
        fake_asr_models.ASRModel.from_pretrained.return_value = fake_nemo_model

        fake_torch = mock.MagicMock()
        mods = {
            "torch": fake_torch,
            "nemo": mock.MagicMock(),
            "nemo.collections": mock.MagicMock(),
            "nemo.collections.asr": mock.MagicMock(),
            "nemo.collections.asr.models": fake_asr_models,
        }
        return mods

    def test_maps_dict_shaped_words(self):
        hyp = SimpleNamespace(
            words=[
                {"word": "hello", "start": 0.0, "end": 0.3, "score": 0.93},
                {"word": "there", "start": 0.4, "end": 0.8, "score": 0.89},
            ],
            word_scores=None,
            confidence=None,
        )
        provider = ParakeetProvider()
        with mock.patch.dict(sys.modules, self._install_fake_nemo(hyp)):
            result = provider.transcribe(self.wav)

        self.assertIsInstance(result, TranscriptResult)
        self.assertEqual([w.word for w in result.words], ["hello", "there"])
        self.assertAlmostEqual(result.words[0].probability, 0.93)
        self.assertEqual(result.metadata["provider"], "parakeet")
        self.assertEqual(result.metadata["model"].startswith("nvidia/"), True)

    def test_hypothesis_level_confidence_used_when_no_word_scores(self):
        hyp = SimpleNamespace(
            words=[{"word": "one", "start": 0.0, "end": 0.2}],
            word_scores=None,
            confidence=0.77,
        )
        provider = ParakeetProvider()
        with mock.patch.dict(sys.modules, self._install_fake_nemo(hyp)):
            result = provider.transcribe(self.wav)
        # Native hypothesis-level confidence preserved for its words.
        self.assertAlmostEqual(result.words[0].probability, 0.77)
        self.assertEqual(result.metadata["probability_source"],
                         "nemo-hypothesis-confidence")

    def test_dependency_error_when_nemo_missing(self):
        clean = {k: None for k in (
            "torch", "nemo", "nemo.collections", "nemo.collections.asr",
            "nemo.collections.asr.models")}
        with mock.patch.dict(sys.modules, clean):
            with self.assertRaises(DependencyError):
                ParakeetProvider().transcribe(self.wav)


class TestCommonShapeAcrossProviders(unittest.TestCase):
    def test_all_providers_return_same_schema(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            wav = make_tiny_wav(Path(tmp.name))

            wx = WhisperXProvider(model="small")
            with mock.patch.dict(sys.modules, TestWhisperXProvider()._install_fake_whisperx()):
                r_wx = wx.transcribe(wav)

            hyp = SimpleNamespace(
                words=[{"word": "hi", "start": 0.0, "end": 0.1}],
                word_scores=None, confidence=None,
            )
            pk = ParakeetProvider()
            with mock.patch.dict(sys.modules, TestParakeetProvider()._install_fake_nemo(hyp)):
                r_pk = pk.transcribe(wav)

            required_meta = {"provider", "model", "device", "language",
                             "audio_duration_seconds",
                             "processing_time_seconds", "word_count"}
            for r in (r_wx, r_pk):
                self.assertIsInstance(r, TranscriptResult)
                self.assertTrue(required_meta.issubset(r.metadata))
                for w in r.words:
                    self.assertEqual(set(w.to_dict()),
                                     {"word", "start", "end", "probability"})
        finally:
            tmp.cleanup()


class TestFasterWhisperProviderValidation(unittest.TestCase):
    def test_wraps_existing_transcriber_config(self):
        p = FasterWhisperProvider(model="tiny", device="cpu",
                                  compute_type="int8", language="en")
        t = p._transcriber_instance()
        self.assertEqual(t.model_size, "tiny")
        self.assertEqual(t.device, "cpu")

    def test_invalid_model_rejected_before_load(self):
        with self.assertRaises(ValidationError):
            FasterWhisperProvider(model="nonexistent-model")


class TestPipelineIntegration(unittest.TestCase):
    def test_run_asr_stage_uses_factory_and_common_output(self):
        import json
        import pipeline

        calls = {}

        class StubResult:
            words = []
            metadata = {"provider": "whisperx", "model": "small",
                        "device": "cpu", "language": "en",
                        "audio_duration_seconds": 1.0,
                        "processing_time_seconds": 2.0,
                        "word_count": 0, "segment_count": 1}

            def to_json_file(self, path):
                calls["transcript_path"] = str(path)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps({"words": []}),
                                      encoding="utf-8")
                return Path(path)

        class StubProvider:
            def __init__(self, **kw):
                calls.update(kw)
                self.model = kw.get("model") or "default"


            def transcribe(self, path):
                calls["input"] = str(path)
                return StubResult()

        tmp = tempfile.TemporaryDirectory()
        try:
            wav = make_tiny_wav(Path(tmp.name))
            out = Path(tmp.name) / "out" / "t.json"
            with mock.patch.object(pipeline, "create_asr_provider",
                                   StubProvider):
                pipeline.run_asr_stage(wav, out, "whisperx", "small",
                                       "cpu", "int8", "en")
            self.assertEqual(calls["provider"], "whisperx")
            self.assertEqual(calls["model"], "small")
            self.assertEqual(calls["device"], "cpu")
            self.assertEqual(calls["input"], str(wav))
            self.assertTrue(out.exists())
        finally:
            tmp.cleanup()

    def test_cli_rejects_unknown_provider(self):
        import asr.__main__ as asr_main

        with self.assertRaises(SystemExit) as ctx:
            asr_main.main(["--input", "x.wav", "--provider", "bogus"])
        self.assertEqual(ctx.exception.code, 2)

    def test_cli_parses_all_providers(self):
        import asr.__main__ as asr_main

        created = {}

        class StubProvider:
            def __init__(self, **kw):
                created.update(kw)
                self.model = kw.get("model") or "default"

            def transcribe(self, path):
                return TranscriptResult(metadata={
                    "provider": "parakeet", "model": "m", "device": "cpu",
                    "language": None, "audio_duration_seconds": 0.0,
                    "processing_time_seconds": 0.0, "word_count": 0,
                }, words=[])

        tmp = tempfile.TemporaryDirectory()
        try:
            out = Path(tmp.name) / "o.json"
            argv = ["--input", str(make_tiny_wav(Path(tmp.name))),
                    "--output", str(out),
                    "--provider", "parakeet",
                    "--language", "none"]
            with mock.patch.object(asr_main, "create_asr_provider",
                                   StubProvider):
                asr_main.main(argv)
            self.assertEqual(created["provider"], "parakeet")
            self.assertIsNone(created["language"])
        finally:
            tmp.cleanup()


@unittest.skipUnless(__import__("importlib.util", fromlist=["util"]).find_spec("whisperx"),
                     "whisperx not installed")
class TestOptionalRealBackends(unittest.TestCase):
    """Optional integration tests — only run when the dependency exists."""

    def test_whisperx_real_transcription(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            wav = make_tiny_wav(Path(tmp.name))
            result = WhisperXProvider(model="tiny").transcribe(wav)
            self.assertIsInstance(result, TranscriptResult)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
