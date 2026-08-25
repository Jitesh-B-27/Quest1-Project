"""V1 end-to-end video-dialogue localization pipeline.

Connects the five existing modules through their Python APIs:

    URL -> download -> audio extraction -> ASR -> target matching -> frame extraction

Python API:
    from pipeline import run_pipeline

    result = run_pipeline("https://...", "My mind rebels at stagnation")

CLI:
    python pipeline.py --url "https://..." --target "My mind rebels at stagnation"

Cache reuse (skips download/audio-extraction stages when media already exists):
    python pipeline.py --video-path video --audio-path audio \
        --target "..." --model-type whisper --model base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yt_dlp

from asr import Transcriber
from asr.exceptions import ASRError
from audio_extractor import AudioExtractionError, extract_audio
from frame_extractor import FrameExtractor, FrameExtractorError
from matcher import DialogueMatcher, MatcherError
from matcher.core import MatchNotFoundError
from matcher.core import Word as MatcherWord

VIDEO_DIR = Path("video")
AUDIO_DIR = Path("audio")
TRANSCRIPT_PATH = Path("transcript/transcript.json")
FRAMES_DIR = Path("frames")
OUTPUT_DIR = Path("output")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
AUDIO_EXTENSIONS = {".wav"}

DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_RETRY_DELAY = 10


class PipelineError(Exception):
    """Raised when a pipeline stage fails. Message identifies the stage."""


@dataclass
class PipelineResult:
    target_text: str
    matched_text: str
    timestamp: float
    frame_number: int
    frame_image_path: str
    similarity: float
    average_word_probability: float
    minimum_word_probability: float
    video_path: str
    audio_path: str
    transcript_path: str
    actual_timestamp: float = 0.0
    fps: float = 0.0
    frame_count: int | None = None
    duration: float = 0.0
    width: int = 0
    height: int = 0
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path = OUTPUT_DIR / "result.json") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# Progress printing helpers
# ---------------------------------------------------------------------------

_WIDTH = 60


def _banner(title: str) -> None:
    print("\n" + "=" * _WIDTH)
    print(title)
    print("=" * _WIDTH)


def _stage_header(index: int, total: int, name: str) -> None:
    print(f"\n[{index}/{total}] {name}")
    print("-" * _WIDTH)


# ---------------------------------------------------------------------------
# Media cache resolution helpers
# ---------------------------------------------------------------------------

def resolve_media_path(value: str | Path, extensions: set[str],
                       label: str) -> Path:
    """Resolve a cached media file or directory to a concrete file path.

    Accepts either a direct file path or a directory (first matching file
    inside, sorted by name). Raises PipelineError when nothing is found.
    """
    p = Path(value)
    if p.is_file():
        return p
    if p.is_dir():
        for candidate in sorted(p.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in extensions:
                return candidate
        raise PipelineError(
            f"No {label} file found in directory '{p}' "
            f"(looked for: {', '.join(sorted(extensions))})"
        )
    raise PipelineError(f"Cached {label} path does not exist: {p.resolve()}")


# ---------------------------------------------------------------------------
# Stage wrappers (one per pipeline stage; each returns the next stage's input)
# ---------------------------------------------------------------------------

def run_download_stage(url: str, proxy: str | None = None) -> Path:
    from video_downloader import download_video

    _stage_header(1, 5, "VIDEO DOWNLOAD")
    print(f"URL: {url}")
    if proxy:
        print(f"Proxy: {proxy}")
    print(f"Status: Downloading (up to {DOWNLOAD_ATTEMPTS} attempts for "
          f"flaky networks)...")
    try:
        video_path = download_video(
            url, VIDEO_DIR, proxy,
            attempts=DOWNLOAD_ATTEMPTS, retry_delay=DOWNLOAD_RETRY_DELAY,
        )
    except yt_dlp.utils.DownloadError as e:
        raise PipelineError(
            f"Video download failed after {DOWNLOAD_ATTEMPTS} attempts. "
            f"If your network intermittently blocks this host, try again or "
            f"use a VPN/proxy (--proxy). Error: {e}"
        ) from e
    print("Status: Complete")
    print(f"Output: {video_path}")
    return video_path


def run_audio_extraction_stage(video_path: Path) -> Path:
    _stage_header(2, 5, "AUDIO EXTRACTION")
    print(f"Input: {video_path}")
    print("Status: Extracting audio...")
    try:
        wav_path = extract_audio(video_path, AUDIO_DIR)
    except AudioExtractionError as e:
        raise PipelineError(f"Audio extraction failed: {e}") from e
    print("Status: Complete")
    print(f"Output: {wav_path}")
    return wav_path


def run_asr_stage(
    wav_path: Path,
    transcript_path: Path,
    model_size: str,
    model_type: str,
    device: str,
    compute_type: str | None,
    language: str | None,
):
    _stage_header(3, 5, "ASR")
    print(f"Input: {wav_path}")
    print(f"Model: {model_size} ({model_type})  Device: {device}  "
          f"Compute type: {compute_type or 'backend default'}")
    print(f"Language: {language or 'auto-detect'}")
    print("Status: Transcribing... (this can take a while)")
    transcriber = Transcriber(
        model_size=model_size, model_type=model_type, device=device,
        compute_type=compute_type, language=language,
    )
    result = transcriber.transcribe(wav_path)
    result.to_json_file(transcript_path)
    m = result.metadata
    print("Status: Complete")
    print(f"Duration: {m['audio_duration_seconds']}s in "
          f"{m['processing_time_seconds']}s processing time")
    print(f"Words: {m['word_count']} ({m['segment_count']} segments)")
    print(f"Output: {transcript_path}")
    return result


def run_matching_stage(target: str, transcript_result):
    _stage_header(4, 5, "TARGET MATCHING")
    print(f'Target: "{target}"')
    print("Status: Searching transcript...")
    matcher = DialogueMatcher()
    # Reuse the in-memory ASR words; no need to reload transcript.json.
    words = [
        MatcherWord(word=w.word, start=w.start, end=w.end,
                    probability=w.probability)
        for w in transcript_result.words
    ]
    try:
        match = matcher.find_best_match_in_words(target, words)
    except MatchNotFoundError as e:
        raise PipelineError(f"Target dialogue not found: {e}") from e
    except MatcherError as e:
        raise PipelineError(f"Target matching failed: {e}") from e
    print("Status: Match found")
    print(f"Matched text: {match.matched_text}")
    print(f"Similarity: {match.text_similarity:.4f}")
    print(f"Avg/Min probability: {match.average_word_probability:.4f} / "
          f"{match.minimum_word_probability:.4f}")
    print(f"Timestamp: {match.start_time:.3f}s - {match.end_time:.3f}s")
    return match


def run_frame_extraction_stage(video_path: Path, start_time: float):
    _stage_header(5, 5, "FRAME EXTRACTION")
    print(f"Input: {video_path}")
    print(f"Timestamp: {start_time:.3f}s")
    print("Status: Extracting frame...")
    extractor = FrameExtractor(output_dir=FRAMES_DIR)
    try:
        frame = extractor.extract_at_timestamp(video_path, start_time)
    except FrameExtractorError as e:
        raise PipelineError(f"Frame extraction failed: {e}") from e
    print("Status: Complete")
    print(f"Frame number: {frame.frame_number} "
          f"(actual timestamp {frame.actual_timestamp:.3f}s)")
    print(f"Image: {frame.image_path}")
    return frame


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    url: str | None = None,
    target: str = "",
    model_size: str = "small",
    model_type: str = "faster-whisper",
    device: str = "cpu",
    compute_type: str | None = None,
    language: str | None = "en",
    proxy: str | None = None,
    save_result: bool = True,
    video_path: str | Path | None = None,
    audio_path: str | Path | None = None,
) -> PipelineResult:
    """Run all five stages in order and return the combined result.

    Media caching: pass ``video_path`` and/or ``audio_path`` (a file or a
    directory containing one) to reuse existing media. When a cached video
    is supplied, stage 1 is skipped; when a cached audio file is supplied,
    stage 2 is skipped. ``url`` is only required when no cached video is
    available.

    Raises:
        PipelineError: If any stage fails (later stages are not executed).
    """
    if not isinstance(target, str) or not target.strip():
        raise PipelineError("Target dialogue must be a non-empty string.")

    # Console may use a legacy codepage (cp1252); never crash on unicode output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

    started = time.monotonic()
    _banner("VIDEO DIALOGUE LOCALIZATION PIPELINE")

    print(f"Model type: {model_type}  Model size: {model_size}")

    cached_video = (
        resolve_media_path(video_path, VIDEO_EXTENSIONS, "video")
        if video_path else None
    )
    cached_audio = (
        resolve_media_path(audio_path, AUDIO_EXTENSIONS, "audio")
        if audio_path else None
    )

    # Stage 1: download (skipped when a cached video is supplied)
    video_path_resolved: Path
    if cached_video is not None:
        _stage_header(1, 5, "VIDEO DOWNLOAD - SKIPPED (CACHE HIT)")
        print(f"Using cached video: {cached_video}")
        video_path_resolved = cached_video
    else:
        if not url or not url.strip():
            raise PipelineError(
                "No video available: provide --url or an existing "
                "--video-path to skip the download stage."
            )
        video_path_resolved = run_download_stage(url.strip(), proxy)

    # Stage 2: audio extraction (skipped when a cached WAV is supplied)
    if cached_audio is not None:
        _stage_header(2, 5, "AUDIO EXTRACTION - SKIPPED (CACHE HIT)")
        print(f"Using cached audio: {cached_audio}")
        wav_path = cached_audio
    else:
        wav_path = run_audio_extraction_stage(video_path_resolved)

    # Stage 3: ASR
    transcript_result = run_asr_stage(
        wav_path, TRANSCRIPT_PATH, model_size, model_type,
        device, compute_type, language,
    )

    # Stage 4: target matching (in-memory ASR words)
    match = run_matching_stage(target.strip(), transcript_result)

    # Stage 5: frame extraction at the matched start time
    frame = run_frame_extraction_stage(video_path_resolved, match.start_time)

    result = PipelineResult(
        target_text=target.strip(),
        matched_text=match.matched_text,
        timestamp=match.start_time,
        frame_number=frame.frame_number,
        frame_image_path=str(frame.image_path),
        similarity=match.text_similarity,
        average_word_probability=match.average_word_probability,
        minimum_word_probability=match.minimum_word_probability,
        video_path=str(video_path_resolved),
        audio_path=str(wav_path),
        transcript_path=str(TRANSCRIPT_PATH),
        actual_timestamp=frame.actual_timestamp,
        fps=frame.fps,
        frame_count=frame.frame_count,
        duration=frame.duration,
        width=frame.width,
        height=frame.height,
        word_count=len(transcript_result.words),
    )

    saved = result.save() if save_result else OUTPUT_DIR / "result.json"

    _banner("PIPELINE COMPLETE")
    print(f"\nTotal time: {time.monotonic() - started:.1f}s")
    print("\nFinal Result:")
    print(f"  Text:       {result.matched_text}")
    print(f"  Timestamp:  {result.timestamp:.3f}s "
          f"(frame {result.frame_number}, actual {result.actual_timestamp:.3f}s)")
    print(f"  Similarity: {result.similarity:.4f}")
    print(f"  Image:      {result.frame_image_path}")
    print(f"  Saved to:   {saved}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python pipeline.py",
        description="End-to-end video-dialogue localization: download a video, "
                    "extract audio, transcribe, find the target dialogue, and "
                    "extract the corresponding frame.",
    )
    parser.add_argument("--url", default=None,
                        help="Video URL to download (optional if --video-path "
                             "points to an existing video)")
    parser.add_argument("--video-path", default=None,
                        help="Optional cached video file or directory; skips "
                             "the download stage (stage 1)")
    parser.add_argument("--audio-path", default=None,
                        help="Optional cached WAV audio file or directory; "
                             "skips the audio extraction stage (stage 2)")
    parser.add_argument("--target", required=True,
                        help="Target dialogue sentence to locate in the video")
    parser.add_argument("--model", default="small",
                        help="ASR model size: tiny, base, small, medium, large "
                             "(default: %(default)s)")
    parser.add_argument("--model-type", default="faster-whisper",
                        choices=["faster-whisper", "whisper"],
                        help="ASR backend (default: %(default)s)")
    parser.add_argument("--language", default="en",
                        help="ASR language code; 'none' for auto-detect "
                             "(default: %(default)s)")
    parser.add_argument("--device", default="cpu",
                        help="ASR device (default: %(default)s)")
    parser.add_argument("--compute-type", default=None,
                        help="ASR compute type, e.g. 'int8' for faster-whisper; "
                             "ignored by the 'whisper' backend "
                             "(default: backend default)")
    parser.add_argument("--proxy", default=None,
                        help="Optional proxy URL for the download stage, "
                             "e.g. socks5://127.0.0.1:1080")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the final result JSON")
    args = parser.parse_args(argv)

    try:
        run_pipeline(
            url=args.url,
            target=args.target,
            model_size=args.model,
            model_type=args.model_type,
            device=args.device,
            compute_type=args.compute_type,
            language=None if args.language.lower() == "none" else args.language,
            proxy=args.proxy,
            save_result=not args.no_save,
            video_path=args.video_path,
            audio_path=args.audio_path,
        )
    except PipelineError as e:
        _stage_failure(str(e))
        sys.exit(1)


def _stage_failure(message: str) -> None:
    print("\n" + "=" * _WIDTH)
    print("PIPELINE FAILED")
    print("=" * _WIDTH)
    print(f"\n{message}", file=sys.stderr)


if __name__ == "__main__":
    main()
