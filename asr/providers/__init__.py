"""ASR backend providers.

Each provider wraps one transcription engine behind the shared
:class:`asr.providers.base.BaseASRProvider` interface.
"""

from __future__ import annotations

from asr.providers.base import BaseASRProvider
from asr.providers.faster_whisper_provider import FasterWhisperProvider
from asr.providers.openai_whisper_provider import OpenAIWhisperProvider
from asr.providers.whisperx_provider import WhisperXProvider

__all__ = [
    "BaseASRProvider",
    "FasterWhisperProvider",
    "OpenAIWhisperProvider",
    "WhisperXProvider",
]
