"""Core candidate-region localization for the V2 coarse-to-fine pipeline.

Responsibilities:
    generate_top_k()       -- rank coarse-transcript windows, keep Top-K (no
                              similarity rejection: the fine stage validates)
    pad_and_merge()        -- pad regions by +/- margin and merge overlaps
    extract_region_wavs()  -- cut region WAVs from the full WAV via FFmpeg
    cleanup_regions()      -- remove temp region WAVs (on success)
    format_timestamp()     -- seconds -> "HH:MM:SS.sss" (display only)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from matcher.core import Word, normalize_text, text_similarity

DEFAULT_TOP_K = 5
DEFAULT_MARGIN_S = 5.0
MAX_OVERLAP_RATIO = 0.5  # NMS: drop candidates overlapping a kept one > 50%
REGIONS_DIR = Path("regions")


class LocalizationError(Exception):
    """Raised when candidate-region generation or extraction fails."""


@dataclass
class CandidateRegion:
    """A time window in the full audio.

    ``global_start`` is the exact value handed to FFmpeg as ``-ss`` when the
    region WAV is extracted; all global timestamps derive from it.
    """

    global_start: float
    global_end: float
    coarse_similarity: float
    source_rank: int          # 1-based rank from the coarse Top-K
    word_indices: tuple[int, int]  # half-open [start, end) into coarse words


# ---------------------------------------------------------------------------
# Coarse Top-K generation (high recall: no similarity threshold here)
# ---------------------------------------------------------------------------

def generate_top_k(
    coarse_words: list[Word],
    target: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[CandidateRegion]:
    """Rank all N-2..N+2 word windows against the target; return Top-K.

    Candidates are ranked purely by fuzzy similarity - no coarse threshold is
    applied. Validation happens later, on the fine transcript.
    """
    target_norm = normalize_text(target)
    if not target_norm:
        raise LocalizationError(
            f"Target normalizes to empty text (only punctuation/whitespace): {target!r}"
        )
    if not coarse_words:
        return []

    n_tokens = len(target_norm.split())
    scored: list[tuple[float, int, int]] = []  # (similarity, start_idx, end_idx)
    for size in range(max(1, n_tokens - 2), n_tokens + 3):
        if size > len(coarse_words):
            break
        for i in range(len(coarse_words) - size + 1):
            window = coarse_words[i:i + size]
            window_norm = " ".join(normalize_text(w.word) for w in window).strip()
            scored.append((text_similarity(target_norm, window_norm), i, i + size))

    # Highest similarity first; ties broken by earlier position.
    scored.sort(key=lambda t: (-t[0], t[1]))

    selected: list[CandidateRegion] = []
    for sim, i, j in scored:
        if len(selected) >= max(1, top_k):
            break
        duplicated = False
        for kept in selected:
            ki, kj = kept.word_indices
            overlap = max(0, min(j, kj) - max(i, ki))
            union = max(j, kj) - min(i, ki)
            if union > 0 and overlap / union > MAX_OVERLAP_RATIO:
                duplicated = True
                break
        if not duplicated:
            selected.append(CandidateRegion(
                global_start=float(coarse_words[i].start),
                global_end=float(coarse_words[j - 1].end),
                coarse_similarity=float(sim),
                source_rank=len(selected) + 1,
                word_indices=(i, j),
            ))
    return selected


# ---------------------------------------------------------------------------
# Padding and merging
# ---------------------------------------------------------------------------

def pad_and_merge(
    regions: list[CandidateRegion],
    margin_s: float = DEFAULT_MARGIN_S,
    audio_duration: float | None = None,
) -> list[CandidateRegion]:
    """Pad each region by +/- ``margin_s`` (clamped to [0, duration]) and
    merge overlapping regions."""
    if not regions:
        return []

    padded: list[CandidateRegion] = []
    for r in regions:
        gs = max(0.0, r.global_start - margin_s)
        ge = r.global_end + margin_s
        if audio_duration is not None and audio_duration > 0:
            ge = min(ge, float(audio_duration))
        padded.append(replace(r, global_start=gs, global_end=ge))

    padded.sort(key=lambda r: r.global_start)

    merged: list[CandidateRegion] = []
    for r in padded:
        if merged and r.global_start <= merged[-1].global_end:
            last = merged[-1]
            merged[-1] = replace(
                last,
                global_end=max(last.global_end, r.global_end),
                # Keep the identity of the better-ranked source region.
                coarse_similarity=max(last.coarse_similarity, r.coarse_similarity),
                source_rank=min(last.source_rank, r.source_rank),
            )
        else:
            merged.append(r)
    return merged


# ---------------------------------------------------------------------------
# Region WAV extraction via FFmpeg
# ---------------------------------------------------------------------------

def extract_region_wavs(
    full_wav_path: str | Path,
    regions: list[CandidateRegion],
    output_dir: str | Path = REGIONS_DIR,
    ffmpeg_path: str | None = None,
    timeout: int = 600,
) -> list[tuple[CandidateRegion, Path]]:
    """Cut one WAV per region out of the full-audio WAV.

    Uses input seeking (``-ss`` before ``-i``), which is sample-accurate for
    PCM input. After spawning each command, ``region.global_start`` is set to
    the exact float parsed back from the ``-ss`` string so the recorded value
    always equals what FFmpeg actually used.

    Returns ordered ``(region, wav_path)`` pairs.
    """
    full_wav = Path(full_wav_path)
    if not full_wav.exists():
        raise LocalizationError(f"Full-audio WAV does not exist: {full_wav.resolve()}")

    if ffmpeg_path is None:
        from audio_extractor import find_ffmpeg

        ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise LocalizationError("FFmpeg not found for region extraction.")

    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LocalizationError(f"Cannot create regions dir '{out_dir}': {e}") from e

    results: list[tuple[CandidateRegion, Path]] = []
    for idx, region in enumerate(regions):
        start_str = f"{region.global_start:.6f}"
        duration = max(0.0, region.global_end - region.global_start)
        wav_path = out_dir / f"region_{idx:02d}.wav"
        cmd = [
            ffmpeg_path, "-y",
            "-ss", start_str,
            "-t", f"{duration:.6f}",
            "-i", str(full_wav),
            str(wav_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise LocalizationError(
                f"FFmpeg timed out extracting region {idx}.") from e
        except OSError as e:
            raise LocalizationError(f"Failed to run FFmpeg: {e}") from e
        if proc.returncode != 0 or not wav_path.exists():
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise LocalizationError(
                f"Region {idx} extraction failed (exit {proc.returncode}): {tail}")

        # The recorded global_start must equal the literal -ss value used.
        region.global_start = float(start_str)
        results.append((region, wav_path))
    return results


def cleanup_regions(output_dir: str | Path = REGIONS_DIR) -> int:
    """Delete temporary region WAVs. Returns how many files were removed."""
    d = Path(output_dir)
    if not d.is_dir():
        return 0
    removed = 0
    for wav in d.glob("region_*.wav"):
        try:
            wav.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# Timestamp formatting (display only - never round internal floats)
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.sss (millisecond precision)."""
    total_ms = round(float(seconds) * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    hh, rem = divmod(total_seconds, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
