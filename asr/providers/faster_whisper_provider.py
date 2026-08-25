"""faster-whisper (CTranslate2) provider.

Word-level timestamps with raw probabilities via CTranslate2 int8 inference
on CPU. This was the original V1 ASR backend.

CPU optimizations applied here:
  * int8 quantized compute (default) - smallest/fastest path on CPU.
  * ``cpu_threads`` pinned to every logical core so CTranslate2 fully
    utilizes the machine during encoder/decoder GEMM work.
  * Greedy decoding (``beam_size=1``) - removes the ~5x beam-search tax of
    the library default with negligible accuracy loss for dialogue
    localization.
  * ``num_workers=1`` - a single tokenization worker avoids redundant
    threads competing with the compute pool.
  * VAD filter skips silence instead of decoding it.
  * Audio handed over as an in-memory float32 array (decoded once via
    ffmpeg through PyAV) rather than re-reading the file inside the lib.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from asr.exceptions import ModelLoadError, TranscriptionError
from asr.models import TranscriptResult, Word
from asr.providers.base import BaseASRProvider, validate_audio_path

DEFAULT_COMPUTE_TYPE = "int8"


def _cpu_threads() -> int:
    return os.cpu_count() or 1


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
                cpu_threads=_cpu_threads(),
                num_workers=1,
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
            import numpy as np
            from faster_whisper.audio import decode_audio

            # Decode exactly once to an in-memory float32 array @ 16 kHz mono.
            audio = decode_audio(str(source), sampling_rate=16000)
            duration = round(float(len(audio)) / 16000.0, 3)

            segments_iter, info = self._model.transcribe(
                audio.astype(np.float32),
                language=self.language,
                vad_filter=True,
                word_timestamps=True,
                beam_size=1,       # greedy: skips the default 5-beam search
                temperature=0.0,   # no multi-temperature retry loops
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
            "audio_duration_seconds": duration,
            "processing_time_seconds": round(elapsed, 3),
            "word_count": len(words),
            "segment_count": segment_count,
            "cpu_threads": _cpu_threads(),
            "beam_size": 1,
        }
        metadata.update(self.metadata_extras())
        return TranscriptResult(metadata=metadata, words=words)
