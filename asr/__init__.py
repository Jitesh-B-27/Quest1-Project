"""ASR stage for the video-dialogue localization pipeline (V1 baseline).

Transcribes an extracted WAV file via faster-whisper on CPU, producing
chronologically-ordered word-level timestamps with raw model probabilities.

Python API:
    from asr import Transcriber

    transcriber = Transcriber()
    result = transcriber.transcribe("audio/audio.wav")
    result.to_json_file("transcript/transcript.json")

CLI:
    python -m asr.transcriber --input audio/audio.wav --output transcript/transcript.json
"""

from typing import Any

__all__ = [
    "ASRError",
    "ModelLoadError",
    "TranscriptionError",
    "ValidationError",
    "Transcriber",
    "TranscriptResult",
]

_LAZY_ATTRS = {
    "ASRError": ("asr.exceptions", "ASRError"),
    "ModelLoadError": ("asr.exceptions", "ModelLoadError"),
    "TranscriptionError": ("asr.exceptions", "TranscriptionError"),
    "ValidationError": ("asr.exceptions", "ValidationError"),
    "Transcriber": ("asr.transcriber", "Transcriber"),
    "TranscriptResult": ("asr.models", "TranscriptResult"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
