"""whisperx provider.

WhisperX = faster-whisper transcription + forced-alignment (wav2vec2) for
sharp word-level timestamps. CPU-optimized settings:
  * int8 CTranslate2 compute (default) and ``threads`` pinned to the core
    count for the underlying faster-whisper engine.
  * Modest batch size (8) - large batches only pay off on GPU.
  * Audio decoded once via ``whisperx.load_audio`` (16 kHz mono float32).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from asr.exceptions import ModelLoadError, TranscriptionError
from asr.models import TranscriptResult, Word
from asr.providers.base import BaseASRProvider, validate_audio_path
from asr.providers.openai_whisper_provider import _ensure_ffmpeg_available

DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BATCH_SIZE = 8  # conservative for CPU; larger batches help GPU only
SAMPLE_RATE = 16000


def _cpu_threads() -> int:
    return os.cpu_count() or 1


class WhisperXProvider(BaseASRProvider):
    name = "whisperx"

    def __init__(
        self,
        model_size: str,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = "en",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__(model_size, device,
                         compute_type or DEFAULT_COMPUTE_TYPE, language)
        self.batch_size = batch_size
        self._model = None
        self._align_model = None
        self._align_metadata: dict | None = None
        self._aligned_language: str | None = None

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._model is not None:
            return
        _ensure_ffmpeg_available()
        try:
            import whisperx

            self._model = whisperx.load_model(
                self.model_size,
                self.device or "cpu",
                compute_type=self.compute_type,
                language=self.language,
                threads=_cpu_threads(),
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load whisperx model '{self.model_size}' "
                f"(device={self.device}, compute_type={self.compute_type}): {e}"
            ) from e

    def _ensure_align_model(self, language_code: str) -> None:
        if self._align_model is not None and self._aligned_language == language_code:
            return
        try:
            import whisperx

            self._align_model, self._align_metadata = (
                whisperx.load_align_model(
                    language_code=language_code, device=self.device or "cpu"
                )
            )
            self._aligned_language = language_code
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load whisperx alignment model for language "
                f"'{language_code}': {e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Transcription
    # ------------------------------------------------------------------ #

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        source = Path(audio_path)
        validate_audio_path(source)
        self.load()

        start = time.monotonic()
        try:
            import numpy as np
            import whisperx

            audio = whisperx.load_audio(str(source))  # float32 @ 16 kHz mono
            duration = round(float(len(audio)) / SAMPLE_RATE, 3)

            asr_result = self._model.transcribe(
                audio.astype(np.float32),
                batch_size=self.batch_size,
            )
            detected_language = asr_result.get("language") or self.language or "en"
            self._ensure_align_model(detected_language)

            aligned = whisperx.align(
                asr_result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.device or "cpu",
                return_char_alignments=False,
            )

            words: list[Word] = []
            segment_count = 0
            for seg in aligned.get("segments", []):
                segment_count += 1
                for w in seg.get("words", []):
                    if w.get("start") is None or w.get("end") is None:
                        continue  # alignment could not place this token
                    words.append(
                        Word(
                            word=str(w.get("word", "")).strip(),
                            start=round(float(w["start"]), 3),
                            end=round(float(w["end"]), 3),
                            probability=round(float(w.get("score", 0.0)), 3),
                        )
                    )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Transcription failed for '{source.name}': {e}") from e

        elapsed = time.monotonic() - start
        metadata = {
            "input_audio": str(source),
            "model": self.model_size,
            "model_type": self.name,
            "device": self.device or "cpu",
            "compute_type": self.compute_type,
            "language": self.language,
            "detected_language": detected_language,
            "audio_duration_seconds": duration,
            "processing_time_seconds": round(elapsed, 3),
            "word_count": len(words),
            "segment_count": segment_count,
            "batch_size": self.batch_size,
            "cpu_threads": _cpu_threads(),
        }
        metadata.update(self.metadata_extras())
        return TranscriptResult(metadata=metadata, words=words)
