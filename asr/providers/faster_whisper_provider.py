"""faster-whisper (CTranslate2) provider.

Word-level timestamps with raw probabilities via CTranslate2 int8 inference
on CPU. This was the original V1 ASR backend.
"""

from __future__ import annotations

import time
from pathlib import Path

from asr.exceptions import ModelLoadError, TranscriptionError
from asr.models import TranscriptResult, Word
from asr.providers.base import BaseASRProvider, validate_audio_path

DEFAULT_COMPUTE_TYPE = "int8"


class FasterWhisperProvider(BaseASRProvider):
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = "en",
    ) -> None:
        super().__init__(model_size, device,
                         compute_type or DEFAULT_COMPUTE_TYPE, language)
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load faster-whisper model '{self.model_size}' "
                f"(device={self.device}, compute_type={self.compute_type}): {e}"
            ) from e

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        source = Path(audio_path)
        validate_audio_path(source)
        self.load()

        start = time.monotonic()
        try:
            segments_iter, info = self._model.transcribe(
                str(source),
                language=self.language,
                vad_filter=True,
                word_timestamps=True,
            )
            words: list[Word] = []
            segment_count = 0
            for seg in segments_iter:
                segment_count += 1
                for w in getattr(seg, "words", None) or []:
                    words.append(
                        Word(
                            word=w.word.strip(),
                            start=round(float(w.start), 3),
                            end=round(float(w.end), 3),
                            probability=round(float(w.probability), 3),
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
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "audio_duration_seconds": round(float(info.duration), 3),
            "processing_time_seconds": round(elapsed, 3),
            "word_count": len(words),
            "segment_count": segment_count,
        }
        metadata.update(self.metadata_extras())
        return TranscriptResult(metadata=metadata, words=words)
