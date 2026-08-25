"""Refine-only forced alignment via WhisperX.

This module performs NO speech recognition. It takes an existing transcript
(words with timings) for a short audio clip and re-aligns it to sharpen the
word-level timestamps. If anything fails, callers fall back to the original
fine-ASR timestamps - alignment must never break the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from asr.exceptions import ASRError
from asr.models import TranscriptResult, Word
from asr.providers.openai_whisper_provider import _ensure_ffmpeg_available

_ALIGN_MODELS: dict[str, tuple] = {}  # language -> (model, metadata)


class AlignmentError(ASRError):
    """Raised when forced alignment cannot be performed at all."""


def _get_align_model(language: str, device: str = "cpu"):
    if language not in _ALIGN_MODELS:
        import whisperx

        _ALIGN_MODELS[language] = whisperx.load_align_model(
            language_code=language, device=device
        )
    return _ALIGN_MODELS[language]


def refine_word_timestamps(
    audio_path: str | Path,
    transcript: TranscriptResult | list[Word],
    language: str = "en",
    device: str = "cpu",
) -> list[Word] | None:
    """Re-align an existing transcript against its own audio clip.

    Args:
        audio_path: The (short) WAV clip the transcript was produced from.
        transcript: A ``TranscriptResult`` or plain list of ``Word``.
        language: ISO code used to pick the alignment model.

    Returns:
        New ``Word`` list with refined timestamps, or ``None`` on any
        failure (callers should fall back to the input timestamps).
    """
    try:
        import whisperx

        _ensure_ffmpeg_available()

        words = (
            transcript.words if isinstance(transcript, TranscriptResult)
            else list(transcript)
        )
        if not words:
            return None

        audio = np.asarray(whisperx.load_audio(str(audio_path)), dtype=np.float32)
        model, metadata = _get_align_model(language or "en", device or "cpu")

        segments = [{
            "start": float(words[0].start),
            "end": max(float(words[-1].end), float(words[0].start)),
            "text": " ".join(w.word for w in words).strip(),
        }]
        result = whisperx.align(
            segments, model, metadata, np.asarray(audio, dtype=np.float32),
            device or "cpu", return_char_alignments=False,
        )

        refined: list[Word] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                refined.append(Word(
                    word=str(w.get("word", "")).strip(),
                    start=round(float(w["start"]), 3),
                    end=round(float(w["end"]), 3),
                    probability=round(float(w.get("score", 0.0)), 3),
                ))
        return refined or None
    except Exception:
        return None  # Graceful degradation: caller keeps fine-ASR timings.


def match_refined_onset(
    matched_words: list[Word],
    refined_words: list[Word],
) -> float | None:
    """Locate the matched phrase inside the refined word stream.

    Matches on the contiguous word-text sequence; among multiple occurrences
    picks the one closest in time to the original match. Returns the refined
    onset in seconds, or ``None`` if the phrase cannot be located.
    """
    if not matched_words or not refined_words:
        return None
    texts = [w.word.strip().lower() for w in matched_words]
    n = len(texts)
    refined_norm = [w.word.strip().lower() for w in refined_words]

    best: tuple[float, float] | None = None  # (time_delta, onset)
    for i in range(len(refined_norm) - n + 1):
        if refined_norm[i:i + n] == texts:
            delta = abs(refined_words[i].start - matched_words[0].start)
            if best is None or delta < best[0]:
                best = (delta, refined_words[i].start)
    return best[1] if best is not None else None
