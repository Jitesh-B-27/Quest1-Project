"""Tests for ASR provider abstraction and pipeline media-cache resolution."""

import unittest

from asr.exceptions import ValidationError
from asr.factory import MODEL_TYPES, create_provider, normalize_model_type
from asr.providers.base import BaseASRProvider
from asr.providers.faster_whisper_provider import FasterWhisperProvider
from asr.providers.openai_whisper_provider import OpenAIWhisperProvider
from asr.providers.whisperx_provider import WhisperXProvider
from asr.transcriber import Transcriber


class TestProviderFactory(unittest.TestCase):
    def test_known_model_types_resolve(self):
        self.assertIs(create_provider("faster-whisper", "base").__class__,
                      FasterWhisperProvider)
        self.assertIs(create_provider("whisper", "base").__class__,
                      OpenAIWhisperProvider)
        self.assertIs(create_provider("whisperx", "base").__class__,
                      WhisperXProvider)

    def test_aliases_normalize(self):
        self.assertEqual(normalize_model_type("whisper"), "whisper")
        self.assertEqual(normalize_model_type("faster_whisper"),
                         "faster-whisper")
        self.assertEqual(normalize_model_type("OpenAI-Whisper"), "whisper")
        self.assertEqual(normalize_model_type("Whisper-X"), "whisperx")

    def test_unknown_model_type_rejected(self):
        with self.assertRaises(ValidationError):
            normalize_model_type("bogus")
        with self.assertRaises(ValidationError):
            create_provider("bogus", "base")

    def test_invalid_model_size_rejected(self):
        with self.assertRaises(ValidationError):
            Transcriber(model_size="giant")

    def test_providers_share_contract(self):
        for cls in MODEL_TYPES.values():
            self.assertTrue(issubclass(cls, BaseASRProvider))
            provider = cls(model_size="tiny")
            self.assertTrue(callable(provider.load))
            self.assertTrue(callable(provider.transcribe))


class TestTranscriberFacade(unittest.TestCase):
    def test_model_type_forwarded_to_provider(self):
        t = Transcriber(model_type="whisper", model_size="base")
        self.assertIsInstance(t.provider, OpenAIWhisperProvider)
        self.assertEqual(t.provider.model_size, "base")

    def test_faster_whisper_default_compute_type(self):
        t = Transcriber(model_type="faster-whisper", model_size="small",
                        compute_type=None)
        self.assertEqual(t.provider.compute_type, "int8")


if __name__ == "__main__":
    unittest.main()
