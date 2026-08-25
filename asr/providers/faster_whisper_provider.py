"""faster-whisper provider — reuses the existing Transcriber unchanged."""

from __future__ import annotations

from asr.exceptions import ModelLoadError, TranscriptionError
from asr.models import Word
from asr.providers.base import ASRProvider


class FasterWhisperProvider(ASRProvider):
    """Existing faster-whisper implementation behind the common interface."""

    provider_name = "faster-whisper"
    default_model = "small"
    valid_models = ("tiny", "base", "small", "medium", "large-v3")

    def __init__(self, model=None, device="cpu", language="en",
                 compute_type="int8") -> None:
        super().__init__(model=model, device=device, language=language,
                         compute_type=compute_type)
        self._transcriber = None

    def _transcriber_instance(self):
        if self._transcriber is None:
            from asr.transcriber import Transcriber

            try:
                self._transcriber = Transcriber(
                    model_size=self.model,
                    device=self.device,
                    compute_type=self.compute_type or "int8",
                    language=self.language,
                )
            except Exception as e:  # invalid constructor args
                raise ModelLoadError(
                    f"Failed to initialize faster-whisper '{self.model}': {e}"
                ) from e
        return self._transcriber

    def _transcribe_words(self, audio_path: str):
        from asr.exceptions import ASRError

        transcriber = self._transcriber_instance()
        # The model loads lazily inside transcribe(); surface load errors.
        try:
            _ = transcriber.model
            result = transcriber.transcribe(audio_path)
        except ASRError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"faster-whisper transcription failed for '{audio_path}': {e}"
            ) from e

        m = result.metadata
        extra = {
            "audio_duration_seconds": m.get("audio_duration_seconds"),
            "segment_count": m.get("segment_count"),
            "compute_type": self.compute_type,
        }
        return list(result.words), extra
