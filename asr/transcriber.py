"""Core Transcriber implementation and CLI entrypoint.

Run as a module:
    python -m asr.transcriber --input audio/audio.wav --output transcript/transcript.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

from asr.exceptions import ModelLoadError, TranscriptionError, ValidationError
from asr.models import TranscriptResult, Word

DEFAULT_INPUT = Path("audio/audio.wav")
DEFAULT_OUTPUT = Path("transcript/transcript.json")
DEFAULT_MODEL_SIZE = "small"
VALID_MODEL_SIZES = ("tiny", "base", "small", "medium")


def _validate_audio_path(path: Path) -> None:
    """Validate the input audio path before processing."""
    if not path.exists():
        raise ValidationError(f"Input file does not exist: {path.resolve()}")
    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {path.resolve()}")
    try:
        if path.stat().st_size == 0:
            raise ValidationError(f"Input file is empty: {path.resolve()}")
    except OSError as e:
        raise ValidationError(f"Cannot read input file '{path}': {e}") from e


class Transcriber:
    """faster-whisper transcription with word-level timestamps.

    Args:
        model_size: Whisper model size (tiny, base, small, medium).
        device: Inference device (e.g. 'cpu').
        compute_type: CTranslate2 compute type (e.g. 'int8' for CPU).
        language: ISO language code to force ('en'); None for auto-detect.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
    ) -> None:
        if model_size not in VALID_MODEL_SIZES:
            raise ValidationError(
                f"Invalid model size '{model_size}'. Valid options: {', '.join(VALID_MODEL_SIZES)}"
            )
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        """Lazily loaded WhisperModel instance."""
        if self._model is None:
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load model '{self.model_size}' "
                    f"(device={self.device}, compute_type={self.compute_type}): {e}"
                ) from e
        return self._model

    def transcribe(self, audio_path: str | Path) -> TranscriptResult:
        """Transcribe audio and return a structured result.

        Raises:
            ValidationError: If the input file is missing/empty/unreadable.
            ModelLoadError: If the model cannot be loaded.
            TranscriptionError: If transcription fails or yields no words.
        """
        source = Path(audio_path)
        _validate_audio_path(source)

        start = time.monotonic()
        try:
            segments_iter, info = self.model.transcribe(
                str(source),
                language=self.language,
                vad_filter=True,
                word_timestamps=True,
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
        except ASRError:
            raise
        except Exception as e:
            raise TranscriptionError(f"Transcription failed for '{source.name}': {e}") from e

        elapsed = time.monotonic() - start
        metadata = {
            "input_audio": str(source),
            "model": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "audio_duration_seconds": round(float(info.duration), 3),
            "processing_time_seconds": round(elapsed, 3),
            "word_count": len(words),
            "segment_count": segment_count,
        }
        return TranscriptResult(metadata=metadata, words=words)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m asr.transcriber",
        description="Transcribe a WAV file with word-level timestamps via faster-whisper.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Input WAV file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE,
                        choices=list(VALID_MODEL_SIZES),
                        help="Model size (default: %(default)s)")
    parser.add_argument("--language", default="en",
                        help="Language code; 'none' for auto-detect (default: %(default)s)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device (default: %(default)s)")
    parser.add_argument("--compute-type", default="int8",
                        help="CTranslate2 compute type (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        transcriber = Transcriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=None if args.language.lower() == "none" else args.language,
        )
        print(f"Loading model '{args.model}' on {args.device} ({args.compute_type})...")
        result = transcriber.transcribe(args.input)
        out_path = result.to_json_file(args.output)
    except (ValidationError, ModelLoadError, TranscriptionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: could not write output file: {e}", file=sys.stderr)
        sys.exit(1)

    m = result.metadata
    print("Transcription complete.")
    print(f"  Input:      {m['input_audio']}")
    print(f"  Output:     {out_path}")
    print(f"  Duration:   {m['audio_duration_seconds']}s in {m['processing_time_seconds']}s")
    print(f"  Segments:   {m['segment_count']}  Words: {m['word_count']}")


if __name__ == "__main__":
    main()
