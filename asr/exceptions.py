"""Custom exceptions for the ASR module."""


class ASRError(Exception):
    """Base exception for all ASR module errors."""


class ValidationError(ASRError):
    """Raised when input validation fails."""


class ModelLoadError(ASRError):
    """Raised when the transcription model fails to load."""


class TranscriptionError(ASRError):
    """Raised when audio processing/transcription fails."""
