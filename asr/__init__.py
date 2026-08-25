"""ASR stage for the video-dialogue localization pipeline.

Supports interchangeable backends (faster-whisper, whisperx, parakeet) via a
common provider interface. All providers accept an audio file path and return
the same ``TranscriptResult`` (metadata + flat word list).

Python API:
    from asr import create_asr_provider

    asr = create_asr_provider(provider="faster-whisper", model="small")
    result = asr.transcribe("audio/audio.wav")
    result.to_json_file("transcript/transcript.json")

CLI:
    python -m asr --input audio/audio.wav --provider faster-whisper --model small
"""

from typing import Any

__all__ = [
    "ASRError",
    "DependencyError",
    "ModelLoadError",
    "TranscriptionError",
    "ValidationError",
    "ASRProvider",
    "FasterWhisperProvider",
    "ParakeetProvider",
    "WhisperXProvider",
    "Transcriber",
    "TranscriptResult",
    "Word",
    "create_asr_provider",
]

_LAZY_ATTRS = {
    # exceptions
    "ASRError": ("asr.exceptions", "ASRError"),
    "DependencyError": ("asr.exceptions", "DependencyError"),
    "ModelLoadError": ("asr.exceptions", "ModelLoadError"),
    "TranscriptionError": ("asr.exceptions", "TranscriptionError"),
    "ValidationError": ("asr.exceptions", "ValidationError"),
    # common interface / providers
    "ASRProvider": ("asr.providers.base", "ASRProvider"),
    "FasterWhisperProvider": ("asr.providers.faster_whisper_provider",
                              "FasterWhisperProvider"),
    "ParakeetProvider": ("asr.providers.parakeet_provider", "ParakeetProvider"),
    "WhisperXProvider": ("asr.providers.whisperx_provider", "WhisperXProvider"),
    # legacy direct faster-whisper API (kept for backwards compatibility)
    "Transcriber": ("asr.transcriber", "Transcriber"),
    # models
    "TranscriptResult": ("asr.models", "TranscriptResult"),
    "Word": ("asr.models", "Word"),
    # factory
    "create_asr_provider": ("asr.factory", "create_asr_provider"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
