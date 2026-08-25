"""Abstract provider interface for the ASR module.

Every ASR backend (faster-whisper, openai-whisper, ...) implements this
interface so that the core ``Transcriber`` contract stays unchanged:
audio file in -> :class:`~asr.models.TranscriptResult` out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from asr.exceptions import ValidationError
from asr.models import TranscriptResult

DEFAULT_DEVICE = "cpu"
DEFAULT_LANGUAGE = "en"


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


class BaseASRProvider(ABC):
    """Common contract for all transcription backends."""

    name: str = "base"

    def __init__(
        self,
        model_size: str,
        device: str = DEFAULT_DEVICE,
        compute_type: str | None = None,
        language: str | None = DEFAULT_LANGUAGE,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

    @abstractmethod
    def load(self) -> None:
        """Load (or lazily prepare) the underlying model."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribe an audio file into a structured result."""

    def metadata_extras(self) -> dict:
        """Backend-specific metadata merged into every result."""
        return {}

    # Shared helpers -----------------------------------------------------

    @staticmethod
    def _new_words_container() -> tuple[list, int]:
        return [], 0
