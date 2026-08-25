"""Core Transcriber implementation and CLI entrypoint.

The public contract is unchanged: an audio file goes in and a
``transcript.json`` comes out. The backend is selected via ``model_type``
(``faster-whisper`` or ``whisper``) and the checkpoint size via
``model_size`` (``tiny``, ``base``, ``small``, ...).

Run as a module:
    python -m asr --input audio/audio.wav --output transcript/transcript.json \
        --model-type whisper --model base
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from asr.exceptions import ASRError, ModelLoadError, TranscriptionError, ValidationError
from asr.factory import MODEL_TYPES, create_provider, normalize_model_type
from asr.models import TranscriptResult
from asr.providers.base import validate_audio_path

DEFAULT_INPUT = Path("audio/audio.wav")
DEFAULT_OUTPUT = Path("transcript/transcript.json")
DEFAULT_MODEL_SIZE = "small"
DEFAULT_MODEL_TYPE = "faster-whisper"
VALID_MODEL_SIZES = ("tiny", "base", "small", "medium", "large")


class Transcriber:
    """Model-agnostic transcription facade.

    Args:
        model_size: Whisper checkpoint size (tiny, base, small, medium, large).
        model_type: Backend to use ('faster-whisper' or 'whisper').
        device: Inference device (e.g. 'cpu').
        compute_type: Backend-specific quantization hint (e.g. 'int8' for
            faster-whisper on CPU); ignored by backends that do not use it.
        language: ISO language code to force ('en'); None for auto-detect.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        model_type: str = DEFAULT_MODEL_TYPE,
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = "en",
    ) -> None:
        if model_size not in VALID_MODEL_SIZES:
            raise ValidationError(
                f"Invalid model size '{model_size}'. "
                f"Valid options: {', '.join(VALID_MODEL_SIZES)}"
            )
        self.model_size = model_size
        # Raises ValidationError for unknown backends.
        self.model_type = normalize_model_type(model_type)
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._provider = create_provider(
            model_type=self.model_type,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )

    @property
    def provider(self):
        """The underlying backend instance (loaded lazily on transcribe)."""
        return self._provider

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribe audio and return a structured result.

        Raises:
            ValidationError: If the input file is missing/empty/unreadable.
            ModelLoadError: If the model cannot be loaded.
            TranscriptionError: If transcription fails.
        """
        return self._provider.transcribe(audio_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m asr",
        description="Transcribe a WAV file with word-level timestamps.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Input audio file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output transcript JSON file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE,
                        choices=sorted(MODEL_TYPES),
                        help="ASR backend (default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE,
                        choices=list(VALID_MODEL_SIZES),
                        help="Model size (default: %(default)s)")
    parser.add_argument("--language", default="en",
                        help="Language code; 'none' for auto-detect (default: %(default)s)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device (default: %(default)s)")
    parser.add_argument("--compute-type", default=None,
                        help="Backend compute type, e.g. 'int8' for faster-whisper "
                             "(default: backend default)")
    args = parser.parse_args(argv)

    try:
        transcriber = Transcriber(
            model_size=args.model,
            model_type=args.model_type,
            device=args.device,
            compute_type=args.compute_type,
            language=None if args.language.lower() == "none" else args.language,
        )
        print(f"Loading model '{args.model}' ({args.model_type}) on "
              f"{args.device}...")
        result = transcriber.transcribe(args.input)
        out_path = result.to_json_file(args.output)
    except ASRError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: could not write output file: {e}", file=sys.stderr)
        sys.exit(1)

    m = result.metadata
    print("Transcription complete.")
    print(f"  Input:      {m['input_audio']}")
    print(f"  Output:     {out_path}")
    print(f"  Model:      {m['model']} ({m['model_type']})")
    print(f"  Duration:   {m['audio_duration_seconds']}s in "
          f"{m['processing_time_seconds']}s")
    print(f"  Segments:   {m['segment_count']}  Words: {m['word_count']}")


if __name__ == "__main__":
    main()
