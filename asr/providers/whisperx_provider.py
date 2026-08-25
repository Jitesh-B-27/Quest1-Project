"""WhisperX provider (transcription + forced alignment, CPU-oriented).

Aligned words carry WhisperX's native alignment ``score`` as the word
probability. If a word has no score, the raw transcription-stage
``probability`` is used when present. No values are fabricated.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from asr.exceptions import (
    DependencyError,
    ModelLoadError,
    TranscriptionError,
)
from asr.models import Word
from asr.providers.base import ASRProvider


class WhisperXProvider(ASRProvider):
    provider_name = "whisperx"
    default_model = "small"
    valid_models = ("tiny", "base", "small", "medium", "large-v3")

    def __init__(self, model=None, device="cpu", language="en",
                 compute_type="int8", batch_size: int = 8) -> None:
        super().__init__(model=model, device=device, language=language,
                         compute_type=compute_type)
        self.batch_size = batch_size
        self._model = None
        self._align_model = None
        self._align_metadata = None

    @staticmethod
    def _ensure_ffmpeg_available() -> None:
        """Expose the bundled FFmpeg to whisperx.

        ``whisperx.load_audio`` shells out to a bare ``ffmpeg`` command; on
        Windows this fails with [WinError 2] unless FFmpeg is on PATH.
        """
        import os

        local_bin = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg" / "bin"
        ffmpeg_exe = local_bin / "ffmpeg.exe"
        if ffmpeg_exe.exists() and str(local_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(local_bin) + os.pathsep + os.environ.get("PATH", "")
        elif not shutil.which("ffmpeg") and not ffmpeg_exe.exists():
            raise DependencyError(
                "FFmpeg is required for whisperx audio loading but was not "
                f"found (looked in '{local_bin}' and on PATH)."
            )

    def _load_transcription_model(self):
        if self._model is None:
            try:
                whisperx = importlib.import_module("whisperx")
            except ImportError as e:
                raise DependencyError(
                    "whisperx is not installed. Install with: "
                    "pip install whisperx  (CPU-only: also ensure torch CPU, "
                    "e.g. pip install torch --index-url https://download.pytorch.org/whl/cpu)"
                ) from e
            try:
                self._model = whisperx.load_model(
                    self.model,
                    self.device,
                    compute_type=self.compute_type or "int8",
                    language=self.language,
                )
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load WhisperX model '{self.model}' on "
                    f"{self.device}: {e}"
                ) from e
        return self._model

    def _load_alignment_model(self, language_code: str | None):
        if self._align_model is None:
            try:
                whisperx = importlib.import_module("whisperx")
                self._align_model, self._align_metadata = (
                    whisperx.load_align_model(
                        language_code=language_code, device=self.device
                    )
                )
            except Exception as e:
                raise TranscriptionError(
                    f"WhisperX alignment model load failed "
                    f"(language={language_code!r}): {e}"
                ) from e
        return self._align_model, self._align_metadata

    @staticmethod
    def _word_confidence(w: dict) -> float:
        """Native confidence from alignment ('score') or ASR ('probability')."""
        for key in ("score", "probability"):
            value = w.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _transcribe_words(self, audio_path: str):
        self._ensure_ffmpeg_available()
        try:
            whisperx = importlib.import_module("whisperx")
        except ImportError as e:  # pragma: no cover - defensive
            raise DependencyError("whisperx is not installed.") from e

        try:
            model = self._load_transcription_model()
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(
                audio, batch_size=self.batch_size, language=self.language
            )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"WhisperX transcription failed for '{audio_path}': {e}"
            ) from e

        detected_language = result.get("language")
        if self.language and detected_language and detected_language != self.language:
            raise TranscriptionError(
                f"Unsupported language/model combination: audio detected as "
                f"'{detected_language}' but provider configured for "
                f"'{self.language}'."
            )

        try:
            align_model, align_metadata = self._load_alignment_model(
                detected_language
            )
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"WhisperX alignment failed: {e}") from e

        words: list[Word] = []
        segment_count = 0
        for seg in aligned.get("segments", []):
            segment_count += 1
            for w in seg.get("words", []):
                text = w.get("word", "").strip()
                if not text:
                    continue
                start = w.get("start")
                end = w.get("end")
                if start is None or end is None:
                    continue
                words.append(self._word(
                    text, start, end, self._word_confidence(w)
                ))

        # Duration estimate from the last aligned timestamp.
        duration = max((w.end for w in words), default=None)
        extra = {
            "audio_duration_seconds": duration,
            "segment_count": segment_count,
            "detected_language": detected_language,
            "compute_type": self.compute_type,
            "probability_source": "whisperx-alignment-score",
        }
        return words, extra
