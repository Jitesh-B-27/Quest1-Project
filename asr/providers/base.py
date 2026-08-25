"""Common ASR provider interface.

All providers accept an audio file path and return the same
``TranscriptResult`` (metadata + flat word list), regardless of backend.
"""

from __future__ import annotations

import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

from asr.exceptions import ValidationError
from asr.models import TranscriptResult, Word


def validate_audio_path(path: Path) -> None:
    """Validate the input audio path before processing."""
    if not path.exists():
        raise ValidationError(f"Input file does not exist: {path.resolve()}")
    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {path.resolve()}")
    try:
        if path.stat().st_size == 0:
            raise ValidationError(f"Input file is empty: {path.resolve()}")
    except OSError as e:
        raise ValidationError(f"Cannot read input file '{path}': {e}") from e


def wav_duration_seconds(path: Path) -> float | None:
    """Best-effort duration from a WAV header; None if unreadable."""
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except Exception:
        return None


class ASRProvider(ABC):
    """Base class for interchangeable ASR backends.

    Subclasses implement ``_transcribe_words(audio_path)``, returning
    ``(words, extra_metadata)``. The public ``transcribe()`` handles
    validation, timing, and metadata so all backends behave identically.
    """

    provider_name: str = "base"
    default_model: str = ""
    valid_models: tuple = ()  # empty tuple = any model string accepted

    def __init__(
        self,
        model: str | None = None,
        device: str = "cpu",
        language: str | None = "en",
        compute_type: str | None = None,
    ) -> None:
        self.model = model or self.default_model
        self.device = device
        self.language = language
        self.compute_type = compute_type

        if self.valid_models and self.model not in self.valid_models:
            raise ValidationError(
                f"Invalid model '{self.model}' for provider '{self.provider_name}'. "
                f"Valid options: {', '.join(self.valid_models)}"
            )

    @abstractmethod
    def _transcribe_words(self, audio_path: str) -> tuple[list[Word], dict]:
        """Run backend inference; return (words, extra_metadata)."""

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        source = Path(audio_path)
        validate_audio_path(source)

        start = time.monotonic()
        words, extra = self._transcribe_words(str(source))
        elapsed = round(time.monotonic() - start, 3)

        duration = extra.pop("audio_duration_seconds", None)
        if duration is None:
            duration = wav_duration_seconds(source)

        metadata = {
            "input_audio": str(source),
            "provider": self.provider_name,
            "model": self.model,
            "device": self.device,
            "language": self.language,
            "audio_duration_seconds": duration,
            "processing_time_seconds": elapsed,
            "word_count": len(words),
            **extra,
        }
        return TranscriptResult(metadata=metadata, words=words)

    @staticmethod
    def _word(text: str, start: float, end: float, probability: float) -> Word:
        return Word(
            word=str(text).strip(),
            start=round(float(start), 3),
            end=round(float(end), 3),
            probability=round(float(probability), 4),
        )
