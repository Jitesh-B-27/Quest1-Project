"""Simple factory for ASR backends."""

from __future__ import annotations

from typing import Any

from asr.exceptions import ValidationError
from asr.providers.base import ASRProvider
from asr.providers.faster_whisper_provider import FasterWhisperProvider
from asr.providers.parakeet_provider import ParakeetProvider
from asr.providers.whisperx_provider import WhisperXProvider

PROVIDERS: dict[str, type[ASRProvider]] = {
    FasterWhisperProvider.provider_name: FasterWhisperProvider,
    WhisperXProvider.provider_name: WhisperXProvider,
    ParakeetProvider.provider_name: ParakeetProvider,
}


def create_asr_provider(
    provider: str = "faster-whisper",
    model: str | None = None,
    device: str = "cpu",
    language: str | None = "en",
    **kwargs: Any,
) -> ASRProvider:
    """Create an ASR provider by name.

    Args:
        provider: One of 'faster-whisper', 'whisperx', 'parakeet'.
        model: Backend model (defaults to the provider's default).
        device: Inference device ('cpu' recommended).
        language: ISO language code, or None for auto-detect.
        **kwargs: Backend-specific options (e.g. compute_type, batch_size).

    Raises:
        ValidationError: If the provider or model is not supported.
    """
    key = (provider or "").strip().lower()
    if key not in PROVIDERS:
        raise ValidationError(
            f"Unsupported provider '{provider}'. "
            f"Valid providers: {', '.join(sorted(PROVIDERS))}"
        )
    return PROVIDERS[key](
        model=model, device=device, language=language, **kwargs
    )
