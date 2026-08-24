"""Extract audio from MP4 files and save it as WAV.

Can be used as an importable module in a pipeline:

    from audio_extractor import extract_audio

    wav_path = extract_audio("video/video1.mp4")

or run directly from the command line:

    python audio_extractor.py [-i INPUT] [-o OUTPUT_DIR]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path("video")
AUDIO_DIR = Path("audio")
LOCAL_FFMPEG = Path(__file__).resolve().parent / "tools" / "ffmpeg" / "bin"


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""


def find_ffmpeg() -> str | None:
    local_exe = LOCAL_FFMPEG / "ffmpeg.exe"
    if local_exe.exists():
        return str(local_exe)
    return shutil.which("ffmpeg")


def find_mp4_in_dir(directory: Path) -> Path:
    """Return the first .mp4 file in the given directory."""
    mp4_files = sorted(directory.glob("*.mp4"))
    if not mp4_files:
        raise AudioExtractionError(
            f"No .mp4 files found in directory: {directory.resolve()}"
        )
    return mp4_files[0]


def extract_audio(
    mp4_path: str | Path,
    output_dir: str | Path = AUDIO_DIR,
    output_name: str | None = None,
    sample_rate: int = 44100,
    channels: int = 2,
) -> Path:
    """Extract the audio track of an MP4 file into a WAV file.

    Args:
        mp4_path: Path to the source .mp4 file.
        output_dir: Directory where the .wav file will be written.
        output_name: Optional name for the .wav file
            (defaults to the source filename with a .wav extension).
        sample_rate: WAV sample rate in Hz.
        channels: Number of audio channels (1 = mono, 2 = stereo).

    Returns:
        Path to the generated .wav file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        AudioExtractionError: If FFmpeg is missing or extraction fails.
    """
    source = Path(mp4_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source.resolve()}")
    if not source.is_file():
        raise AudioExtractionError(f"Source path is not a file: {source.resolve()}")

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise AudioExtractionError(
            "ffmpeg not found. Expected bundled copy at "
            f"'{LOCAL_FFMPEG}' or on system PATH."
        )

    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise AudioExtractionError(f"Could not create output directory '{out_dir}': {e}") from e

    wav_path = out_dir / (output_name or f"{source.stem}.wav")
    if not wav_path.suffix.lower() == ".wav":
        wav_path = wav_path.with_suffix(".wav")

    cmd = [
        ffmpeg_path,
        "-y",                      # overwrite existing output
        "-i", str(source),
        "-vn",                     # drop video stream
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(wav_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except subprocess.TimeoutExpired as e:
        raise AudioExtractionError(f"Audio extraction timed out for '{source.name}'.") from e
    except OSError as e:
        raise AudioExtractionError(f"Failed to run ffmpeg: {e}") from e

    if result.returncode != 0 or not wav_path.exists():
        stderr_tail = (result.stderr or "").strip().splitlines()[-5:]
        raise AudioExtractionError(
            f"Audio extraction failed for '{source.name}' "
            f"(exit code {result.returncode}):\n" + "\n".join(stderr_tail)
        )

    return wav_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the audio track from an MP4 file and save it as WAV."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="Path to the source .mp4 file (default: first .mp4 found in ./video)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=AUDIO_DIR,
        help="Directory to save the .wav file (default: ./audio)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional name for the .wav file (default: same name as source)",
    )
    args = parser.parse_args()

    try:
        source = args.input or find_mp4_in_dir(VIDEO_DIR)
        wav_path = extract_audio(source, args.output_dir, args.output_name)
    except (FileNotFoundError, AudioExtractionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    size_mb = wav_path.stat().st_size / (1024 * 1024)
    print(f"Extracted audio: {source}")
    print(f"Saved to: {wav_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
