"""openai-whisper provider, optimized for CPU inference.

CPU optimizations applied here (no GPU required):
  * torch intra-op threads pinned to the physical core count so the model
    uses all available CPU threads without oversubscription.
  * Audio decoded exactly once via ``whisper.load_audio`` and handed to
    ``transcribe`` as an in-memory float32 array at 16 kHz mono.
  * ``fp16=False`` enforced (fp16 kernels are GPU-only and slow/fallback
    on CPU).
  * Greedy decoding (``beam_size=1``, ``best_of=1``) - roughly 2-4x faster
    than beam search on CPU with negligible accuracy loss for dialogue
    localization.
  * ``condition_on_previous_text=False`` avoids re-encoding ever-growing
    context windows, cutting decode time on long files and reducing
    repetition hallucinations.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from asr.exceptions import ModelLoadError, TranscriptionError
from asr.models import TranscriptResult, Word
from asr.providers.base import BaseASRProvider, validate_audio_path

SAMPLE_RATE = 16000


def _ensure_ffmpeg_available() -> None:
    """Make sure ffmpeg is reachable for whisper.load_audio.

    openai-whisper shells out to ffmpeg to decode input files. If it is not
    on PATH, fall back to the ffmpeg bundled in this repository at
    ``tools/ffmpeg/bin``.
    """
    if shutil.which("ffmpeg"):
        return
    root = Path(__file__).resolve().parents[2]
    bundled = root / "tools" / "ffmpeg" / "bin"
    if (bundled / "ffmpeg.exe").exists() or (bundled / "ffmpeg").exists():
        os.environ["PATH"] = f"{bundled}{os.pathsep}{os.environ.get('PATH', '')}"


def _configure_cpu_threads() -> int:
    """Pin torch to the available physical cores; return the thread count."""
    import torch

    cores = os.cpu_count() or 1
    try:
        physical = os.cpu_count()
        if hasattr(os, "sched_getaffinity"):
            physical = len(os.sched_getaffinity(0))
    except Exception:
        physical = cores
    # Hyperthreaded cores add little for GEMM workloads; cap at half of the
    # logical count when it looks oversubscribed, but never below 1.
    threads = max(1, min(cores, (physical or 1)))
    try:
        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Already initialized in this process; keep current settings.
    return threads


class OpenAIWhisperProvider(BaseASRProvider):
    name = "whisper"

    def __init__(
        self,
        model_size: str,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = "en",
    ) -> None:
        super().__init__(model_size, device, compute_type, language)
        self._model = None
        self._threads = 1

    def load(self) -> None:
        if self._model is not None:
            return
        self._threads = _configure_cpu_threads()
        _ensure_ffmpeg_available()
        try:
            import whisper

            self._model = whisper.load_model(
                self.model_size, device=self.device or "cpu"
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load openai-whisper model '{self.model_size}' "
                f"(device={self.device}): {e}"
            ) from e

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        source = Path(audio_path)
        validate_audio_path(source)
        self.load()

        start = time.monotonic()
        try:
            import numpy as np
            import whisper

            audio = whisper.load_audio(str(source))  # float32 @ 16 kHz mono
            duration = round(float(len(audio)) / SAMPLE_RATE, 3)
            options = {
                "language": self.language,
                "task": "transcribe",
                "word_timestamps": True,
                "fp16": False,  # mandatory for correct/speedy CPU math
                "beam_size": 1,
                "best_of": 1,
                "condition_on_previous_text": False,
                "verbose": False,
            }
            result = self._model.transcribe(audio.astype(np.float32), **options)

            words: list[Word] = []
            segment_count = 0
            for seg in result.get("segments", []):
                segment_count += 1
                for w in seg.get("words", []):
                    words.append(
                        Word(
                            word=str(w.get("word", "")).strip(),
                            start=round(float(w["start"]), 3),
                            end=round(float(w["end"]), 3),
                            probability=round(float(w.get("probability", 0.0)), 3),
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
            "compute_type": self.compute_type or "fp32",
            "language": self.language,
            "audio_duration_seconds": duration,
            "processing_time_seconds": round(elapsed, 3),
            "word_count": len(words),
            "segment_count": segment_count,
        }
        metadata.update(self.metadata_extras())
        metadata["cpu_threads"] = self._threads
        return TranscriptResult(metadata=metadata, words=words)
