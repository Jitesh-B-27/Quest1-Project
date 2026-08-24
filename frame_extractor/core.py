"""Core logic for the V1 frame extractor.

Responsibilities kept separable for future optimization:
    get_video_metadata()         -- video validation / metadata
    timestamp_to_frame_number()  -- timestamp -> frame selection (isolated;
                                    replace here for VFR/accuracy work)
    _extract_frame_image()       -- frame extraction via FFmpeg
    FrameResult                  -- result construction
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("frames")
LOCAL_FFMPEG_BIN = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg" / "bin"


class FrameExtractorError(Exception):
    """Base exception for the frame extractor module."""


class ValidationError(FrameExtractorError):
    """Raised for bad inputs: missing files, invalid timestamps, etc."""


class MetadataError(FrameExtractorError):
    """Raised when video metadata is missing, invalid, or unreadable."""


class ExtractionError(FrameExtractorError):
    """Raised when the frame cannot be extracted or written."""


class TimestampToFrameNumberError(FrameExtractorError):
    """Raised when timestamp -> frame number conversion fails."""


def find_tool(name: str) -> str:
    """Locate a tool (ffmpeg/ffprobe) in the bundled dir or on PATH."""
    found = shutil.which(name, path=str(LOCAL_FFMPEG_BIN))
    if found:
        return str(Path(found).resolve())
    found = shutil.which(name)
    if not found:
        raise ExtractionError(
            f"'{name}' not found. Expected bundled copy at '{LOCAL_FFMPEG_BIN}' "
            f"or on system PATH."
        )
    return found


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int | None
    duration: float
    width: int
    height: int


@dataclass
class FrameResult:
    timestamp: float
    frame_number: int
    actual_timestamp: float
    image_path: str
    fps: float
    frame_count: int | None
    duration: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Video validation / metadata
# ---------------------------------------------------------------------------

def get_video_metadata(video_path: Path) -> VideoMetadata:
    """Probe the video with ffprobe and validate required metadata."""
    if not video_path.exists():
        raise ValidationError(f"Video file does not exist: {video_path.resolve()}")
    if not video_path.is_file():
        raise ValidationError(f"Video path is not a file: {video_path.resolve()}")
    if video_path.stat().st_size == 0:
        raise ValidationError(f"Video file is empty: {video_path.resolve()}")

    ffprobe = find_tool("ffprobe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired as e:
        raise MetadataError(f"ffprobe timed out on '{video_path.name}'.") from e
    except OSError as e:
        raise MetadataError(f"Failed to run ffprobe: {e}") from e

    if proc.returncode != 0 or not proc.stdout.strip():
        raise MetadataError(
            f"Video could not be opened or has no readable metadata: "
            f"'{video_path.name}'"
        )

    import json
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise MetadataError(f"Could not parse ffprobe output: {e}") from e

    video_streams = [s for s in data.get("streams", [])
                     if s.get("codec_type") == "video"]
    if not video_streams:
        raise MetadataError(f"No video stream found in '{video_path.name}'.")
    stream = video_streams[0]

    try:
        fps = float(Fraction(stream["r_frame_rate"]))
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise MetadataError(f"Invalid video metadata (fps/resolution): {e}") from e
    if fps <= 0:
        raise MetadataError(f"Invalid FPS value: {fps}")

    duration_raw = stream.get("duration") or data.get("format", {}).get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError) as e:
        raise MetadataError("Missing or invalid duration in video metadata.") from e
    if duration <= 0:
        raise MetadataError(f"Invalid duration value: {duration}")

    frame_count_raw = stream.get("nb_frames")
    frame_count: int | None = None
    if isinstance(frame_count_raw, str) and frame_count_raw.isdigit():
        frame_count = int(frame_count_raw)
    else:
        # Some containers omit nb_frames; estimate from timing info.
        estimated = round(duration * fps)
        frame_count = estimated if estimated > 0 else None

    return VideoMetadata(fps=fps, frame_count=frame_count,
                         duration=duration, width=width, height=height)


# ---------------------------------------------------------------------------
# Timestamp -> frame selection (V1: CFR assumption, isolated for replacement)
# ---------------------------------------------------------------------------

def timestamp_to_frame_number(timestamp: float, fps: float) -> int:
    """Map a timestamp to its frame index.

    V1 rule for constant-frame-rate video:
        frame_number = floor(timestamp * fps)

    Replace this function when adding accurate seeking, VFR support,
    or timestamp offsets.
    """
    if fps <= 0:
        raise TimestampToFrameNumberError(f"FPS must be positive, got {fps}.")
    return int(timestamp * fps)


def frame_number_to_timestamp(frame_number: int, fps: float) -> float:
    """Inverse mapping: presentation time of a frame index."""
    if fps <= 0:
        raise TimestampToFrameNumberError(f"FPS must be positive, got {fps}.")
    return frame_number / fps


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frame_image(ffmpeg: str, video_path: Path,
                         seek_seconds: float, image_path: Path) -> None:
    """Seek to seek_seconds and save the first decoded frame as JPEG."""
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{seek_seconds:.6f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(image_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
    except subprocess.TimeoutExpired as e:
        raise ExtractionError(f"Frame extraction timed out for '{video_path.name}'.") from e
    except OSError as e:
        raise ExtractionError(f"Failed to run ffmpeg: {e}") from e

    if proc.returncode != 0 or not image_path.exists() or image_path.stat().st_size == 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise ExtractionError(
            f"Frame could not be extracted from '{video_path.name}' at "
            f"{seek_seconds:.3f}s:\n" + "\n".join(stderr_tail)
        )


# ---------------------------------------------------------------------------
# FrameExtractor
# ---------------------------------------------------------------------------

class FrameExtractor:
    """Extract the video frame corresponding to a dialogue timestamp."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)

    def extract_at_timestamp(
        self,
        video_path: str | Path,
        timestamp: float,
        output_dir: str | Path | None = None,
        output_name: str | None = None,
    ) -> FrameResult:
        """Extract and save the frame at ``timestamp``; returns FrameResult."""
        video_path = Path(video_path)
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir

        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise ValidationError(
                f"Timestamp must be a number in seconds, got {timestamp!r}."
            )
        timestamp = float(timestamp)

        metadata = get_video_metadata(video_path)

        if timestamp < 0:
            raise ValidationError(
                f"Timestamp must be non-negative, got {timestamp:.3f}s."
            )
        if timestamp > metadata.duration + 1e-6:
            raise ValidationError(
                f"Timestamp {timestamp:.3f}s is beyond video duration "
                f"({metadata.duration:.3f}s)."
            )

        frame_number = timestamp_to_frame_number(timestamp, metadata.fps)
        if metadata.frame_count is not None:
            frame_number = min(frame_number, metadata.frame_count - 1)
        actual_timestamp = frame_number_to_timestamp(frame_number, metadata.fps)

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ExtractionError(
                f"Could not create output directory '{out_dir}': {e}"
            ) from e

        image_path = out_dir / (output_name or f"frame_{frame_number}.jpg")

        ffmpeg = find_tool("ffmpeg")
        _extract_frame_image(ffmpeg, video_path, actual_timestamp, image_path)

        return FrameResult(
            timestamp=round(timestamp, 3),
            frame_number=frame_number,
            actual_timestamp=round(actual_timestamp, 3),
            image_path=str(image_path),
            fps=metadata.fps,
            frame_count=metadata.frame_count,
            duration=metadata.duration,
            width=metadata.width,
            height=metadata.height,
        )
