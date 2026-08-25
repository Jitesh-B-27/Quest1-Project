"""V1 end-to-end video-dialogue localization pipeline.

Connects the five existing modules through their Python APIs:

    URL -> download -> audio extraction -> ASR -> target matching -> frame extraction

Python API:
    from pipeline import run_pipeline

    result = run_pipeline("https://...", "My mind rebels at stagnation")

CLI:
    python pipeline.py --url "https://..." --target "My mind rebels at stagnation"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yt_dlp

from asr import create_asr_provider
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
    provider_name: str,
    model_size: str | None,
    device: str,
    compute_type: str,
    language: str | None,
):
    _stage_header(3, 5, "ASR")
    print(f"Input: {wav_path}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Provider: {provider_name}")
    print("Status: Transcribing... (this can take a while)")
    try:
        asr_provider = create_asr_provider(
            provider=provider_name, model=model_size, device=device,
            language=language, compute_type=compute_type,
        )
        print(f"Model: {asr_provider.model}  Device: {device}  "
              f"Compute type: {compute_type}")
        result = asr_provider.transcribe(wav_path)
    except ASRError as e:
        raise PipelineError(f"ASR failed: {e}") from e
    result.to_json_file(transcript_path)
    m = result.metadata
    print("Status: Complete")
    print(f"Duration: {m.get('audio_duration_seconds')}s in "
          f"{m['processing_time_seconds']}s processing time")
    print(f"Words: {m['word_count']} ({m.get('segment_count')} segments)")
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
    video_path: str | None = None,
    audio_path: str | None = None,
    asr_provider: str = "faster-whisper",
    model_size: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    proxy: str | None = None,
    save_result: bool = True,
) -> PipelineResult:
    """Run all five stages in order and return the combined result.

    Raises:
        PipelineError: If any stage fails (later stages are not executed).
    """
    if not url and not video_path:
        raise PipelineError("Either a video URL (--url) or a local video "
                            "file (--video) must be provided.")
    if url and video_path:
        raise PipelineError("Provide either --url or --video, not both.")
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

    # Stage 1: download (skipped when a local video is supplied)
    if video_path:
        video = Path(video_path)
        if not video.exists():
            raise PipelineError(
                f"Local video file does not exist: {video.resolve()}")
        _stage_header(1, 5, "VIDEO DOWNLOAD")
        print("Status: Skipped (using local video)")
        print(f"Input: {video}")
    else:
        video = run_download_stage(url.strip(), proxy)

    # Stage 2: audio extraction (skipped when an extracted WAV is supplied)
    if audio_path:
        wav_path = Path(audio_path)
        if not wav_path.exists():
            raise PipelineError(
                f"Local audio file does not exist: {wav_path.resolve()}")
        _stage_header(2, 5, "AUDIO EXTRACTION")
        print("Status: Skipped (using existing WAV)")
        print(f"Input: {wav_path}")
    else:
        wav_path = run_audio_extraction_stage(video)

    # Stage 3: ASR
    transcript_result = run_asr_stage(
        wav_path, TRANSCRIPT_PATH, asr_provider, model_size, device,
        compute_type, language,
    )

    # Stage 4: target matching (in-memory ASR words)
    match = run_matching_stage(target.strip(), transcript_result)

    # Stage 5: frame extraction at the matched start time
    frame = run_frame_extraction_stage(video, match.start_time)

    result = PipelineResult(
        target_text=target.strip(),
        matched_text=match.matched_text,
        timestamp=match.start_time,
        frame_number=frame.frame_number,
        frame_image_path=str(frame.image_path),
        similarity=match.text_similarity,
        average_word_probability=match.average_word_probability,
        minimum_word_probability=match.minimum_word_probability,
        video_path=str(video),
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
                        help="Video URL to download (omit when using --video)")
    parser.add_argument("--target", required=True,
                        help="Target dialogue sentence to locate in the video")
    parser.add_argument("--video", default=None,
                        help="Path to a local video file; skips the download stage")
    parser.add_argument("--audio", default=None,
                        help="Path to an existing WAV file; skips the audio "
                             "extraction stage")
    parser.add_argument("--asr", default="faster-whisper",
                        choices=["faster-whisper", "whisperx", "parakeet"],
                        help="ASR backend provider (default: %(default)s)")
    parser.add_argument("--model", default=None,
                        help="ASR model. Defaults per provider: whisper sizes "
                             "(tiny/base/small/medium/large-v3) for "
                             "faster-whisper and whisperx; a HuggingFace name "
                             "(e.g. nvidia/parakeet-tdt-0.6b-v3) for parakeet")
    parser.add_argument("--language", default="en",
                        help="ASR language code; 'none' for auto-detect "
                             "(default: %(default)s)")
    parser.add_argument("--device", default="cpu",
                        help="ASR device (default: %(default)s)")
    parser.add_argument("--compute-type", default="int8",
                        help="ASR compute type (default: %(default)s)")
    parser.add_argument("--proxy", default=None,
                        help="Optional proxy URL for the download stage, "
                             "e.g. socks5://127.0.0.1:1080")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the final result JSON")
    args = parser.parse_args(argv)

    if not args.url and not args.video:
        parser.error("either --url or --video is required")
    if args.url and args.video:
        parser.error("--url and --video are mutually exclusive")

    try:
        run_pipeline(
            url=args.url,
            video_path=args.video,
            audio_path=args.audio,
            target=args.target,
            asr_provider=args.asr,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=None if args.language.lower() == "none" else args.language,
            proxy=args.proxy,
            save_result=not args.no_save,
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
