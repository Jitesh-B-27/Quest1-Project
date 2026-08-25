"""ASR stage for the video-dialogue localization pipeline.

Transcribes an extracted audio file with word-level timestamps and raw
model probabilities. The backend is abstracted behind a provider factory:

  * ``faster-whisper`` - CTranslate2 int8 CPU inference (original backend)
  * ``whisper``        - openai-whisper, CPU-optimized (threads pinned,
                         greedy decoding, fp32 math)

Python API:
    from asr import Transcriber

    transcriber = Transcriber(model_type="whisper", model_size="base")
    result = transcriber.transcribe("audio/audio.wav")
    result.to_json_file("transcript/transcript.json")

CLI:
    python -m asr --input audio/audio.wav --model-type whisper --model base
"""

from typing import Any

__all__ = [
    "ASRError",
    "ModelLoadError",
    "TranscriptionError",
    "ValidationError",
    "Transcriber",
    "TranscriptResult",
    "create_provider",
]

_LAZY_ATTRS = {
    "ASRError": ("asr.exceptions", "ASRError"),
    "ModelLoadError": ("asr.exceptions", "ModelLoadError"),
    "TranscriptionError": ("asr.exceptions", "TranscriptionError"),
    "ValidationError": ("asr.exceptions", "ValidationError"),
    "Transcriber": ("asr.transcriber", "Transcriber"),
    "TranscriptResult": ("asr.models", "TranscriptResult"),
    "create_provider": ("asr.factory", "create_provider"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
