"""CLI entrypoint for the ASR package (all providers).

Usage:
    python -m asr --input audio/video1.wav --provider faster-whisper --model small
    python -m asr --input audio/video1.wav --provider whisperx --model small
    python -m asr --input audio/video1.wav --provider parakeet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asr.exceptions import ASRError, DependencyError
from asr.factory import PROVIDERS, create_asr_provider

DEFAULT_INPUT = Path("audio/audio.wav")
DEFAULT_OUTPUT = Path("transcript/transcript.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m asr",
        description="Transcribe a WAV file with word-level timestamps using "
                    "a pluggable ASR backend.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Input WAV file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--provider", default="faster-whisper",
                        choices=sorted(PROVIDERS),
                        help="ASR backend (default: %(default)s)")
    parser.add_argument("--model", default=None,
                        help="Backend model (defaults to the provider's "
                             "default model)")
    parser.add_argument("--language", default="en",
                        help="Language code; 'none' for auto-detect "
                             "(default: %(default)s)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device (default: %(default)s)")
    parser.add_argument("--compute-type", default=None,
                        help="Quantization/compute type where supported, "
                             "e.g. int8 (whisper backends only)")
    args = parser.parse_args(argv)

    try:
        provider = create_asr_provider(
            provider=args.provider,
            model=args.model,
            device=args.device,
            language=None if (args.language or "").lower() == "none"
            else args.language,
            compute_type=args.compute_type,
        )
        print(f"Provider: {args.provider}  Model: {provider.model}  "
              f"Device: {args.device}")
        print("Status: Transcribing... (this can take a while)")
        result = provider.transcribe(args.input)
        out_path = result.to_json_file(args.output)
    except DependencyError as e:
        print(f"Error: missing dependency: {e}", file=sys.stderr)
        sys.exit(1)
    except ASRError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: could not write output file: {e}", file=sys.stderr)
        sys.exit(1)

    m = result.metadata
    print("Transcription complete.")
    print(f"  Input:      {m.get('input_audio', args.input)}")
    print(f"  Output:     {out_path}")
    print(f"  Duration:   {m.get('audio_duration_seconds')}s in "
          f"{m.get('processing_time_seconds')}s")
    print(f"  Words:      {m.get('word_count')}")


if __name__ == "__main__":
    main()
