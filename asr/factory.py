"""Provider factory: map a model_type string to a concrete ASR backend."""

from __future__ import annotations

from asr.exceptions import ValidationError
from asr.providers.base import BaseASRProvider
from asr.providers.faster_whisper_provider import FasterWhisperProvider
from asr.providers.openai_whisper_provider import OpenAIWhisperProvider
from asr.providers.whisperx_provider import WhisperXProvider

MODEL_TYPES: dict[str, type[BaseASRProvider]] = {
    "faster-whisper": FasterWhisperProvider,
    "whisper": OpenAIWhisperProvider,
    "whisperx": WhisperXProvider,
}

# Aliases accepted on the CLI for convenience.
_MODEL_TYPE_ALIASES = {
    "faster_whisper": "faster-whisper",
    "fasterwhisper": "faster-whisper",
    "openai-whisper": "whisper",
    "openai_whisper": "whisper",
    "whisper-x": "whisperx",
}


def normalize_model_type(model_type: str) -> str:
    key = (model_type or "").strip().lower()
    key = _MODEL_TYPE_ALIASES.get(key, key)
    if key not in MODEL_TYPES:
        raise ValidationError(
            f"Invalid model type '{model_type}'. "
            f"Valid options: {', '.join(sorted(MODEL_TYPES))}"
        )
    return key


def create_provider(
    model_type: str,
    model_size: str,
    device: str = "cpu",
    compute_type: str | None = None,
    language: str | None = "en",
) -> BaseASRProvider:
    """Instantiate the ASR backend matching ``model_type``.

    Raises:
        ValidationError: If the model type is unknown.
    """
    normalized = normalize_model_type(model_type)
    return MODEL_TYPES[normalized](
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
