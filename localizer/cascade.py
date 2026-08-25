"""Coarse-to-fine cascade for the V2 localization pipeline.

Fallback ladder (each tier reuses the SAME full-audio WAV; media is never
re-downloaded or re-extracted):

    T0: tiny  full audio -> Top-K regions -> small fine validation
    T1: base  full audio -> Top-K regions -> small fine validation
    T2: small full audio + matcher on the full WAV (V1 behavior, in-process)
    T3: MatchNotFoundError

The coarse stage is purely high-recall (no similarity rejection); the fine
stage similarity is the actual validation gate. Forced alignment only refines
the winning region's timestamps and never fails the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from asr import Transcriber
from asr.aligner import match_refined_onset, refine_word_timestamps
from asr.models import TranscriptResult, Word
from localizer.core import (
    DEFAULT_MARGIN_S,
    DEFAULT_TOP_K,
    REGIONS_DIR,
    CandidateRegion,
    LocalizationError,
    extract_region_wavs,
    generate_top_k,
    pad_and_merge,
)
from matcher.core import DialogueMatcher, MatchNotFoundError, MatcherError

LogFn = Callable[[str], None]


@dataclass
class CascadeResult:
    """Outcome of the coarse-to-fine cascade."""

    global_timestamp: float          # precise onset in FULL-audio seconds
    matched_text: str
    similarity: float
    average_word_probability: float
    minimum_word_probability: float
    fallback_tier: str               # e.g. "tiny->small", "small-full"
    coarse_model: str
    fine_model: str
    aligned: bool                    # True if forced alignment refined the onset
    region_count: int
    timings: dict = field(default_factory=dict)
    candidates_summary: list[dict] = field(default_factory=list)


def _transcribe(model_size: str, wav_path: Path, language: str | None,
                device: str, compute_type: str | None) -> TranscriptResult:
    return Transcriber(
        model_size=model_size,
        model_type="faster-whisper",
        device=device,
        compute_type=compute_type,
        language=language,
    ).transcribe(wav_path)


def _validate_regions(
    wav_regions: list[tuple[CandidateRegion, Path]],
    target: str,
    fine_model: str,
    min_similarity: float,
    language: str | None,
    device: str,
    compute_type: str | None,
    log: LogFn | None,
) -> tuple | None:
    """Run fine ASR + matching per region; return the best validated hit."""
    matcher = DialogueMatcher(min_similarity=min_similarity)
    best: tuple | None = None  # (score_key, match, fine_result, region, wav)

    for region, wav_path in wav_regions:
        fine = _transcribe(fine_model, wav_path, language, device, compute_type)
        if not fine.words:
            continue
        try:
            match = matcher.find_best_match_in_words(target, fine.words)
        except (MatchNotFoundError, MatcherError):
            continue  # this region failed validation; try the next one

        score_key = (
            match.text_similarity,
            match.average_word_probability,
            match.minimum_word_probability,
        )
        if log:
            log(f"  Region [{region.global_start:.3f}s - {region.global_end:.3f}s]: "
                f"sim={match.text_similarity:.4f} "
                f"avg_p={match.average_word_probability:.4f}")
        if best is None or score_key > best[0]:
            best = (score_key, match, fine, region, wav_path)

    return best


def run_cascade(
    wav_path: str | Path,
    target: str,
    *,
    coarse_tiers: tuple[str, ...] = ("tiny", "base"),
    fine_model: str = "small",
    top_k: int = DEFAULT_TOP_K,
    region_margin_s: float = DEFAULT_MARGIN_S,
    min_similarity: float = 0.7,
    language: str | None = "en",
    device: str = "cpu",
    compute_type: str | None = None,
    regions_dir: str | Path = REGIONS_DIR,
    log: LogFn | None = None,
) -> CascadeResult:
    """Locate ``target`` inside the full-audio WAV; returns a CascadeResult.

    Raises:
        LocalizationError: On malformed input or FFmpeg extraction failure.
        MatchNotFoundError: If every tier fails to validate the target.
    """
    if not isinstance(target, str) or not target.strip():
        raise MatcherError("Target text must be a non-empty string.")
    wav_path = Path(wav_path)
    timings: dict[str, float] = {}

    def say(msg: str) -> None:
        if log:
            log(msg)

    for tier_index, coarse_model in enumerate(coarse_tiers):
        tier_name = f"{coarse_model}->{fine_model}"

        t0 = time.monotonic()
        coarse = _transcribe(coarse_model, wav_path, language, device, compute_type)
        timings[f"coarse_asr_{coarse_model}_s"] = round(time.monotonic() - t0, 3)
        say(f"Coarse ASR ({coarse_model}): {len(coarse.words)} words "
            f"in {timings[f'coarse_asr_{coarse_model}_s']}s")

        if not coarse.words:
            continue

        duration = coarse.metadata.get("audio_duration_seconds")
        regions = generate_top_k(coarse.words, target, top_k=top_k)
        if not regions:
            continue
        regions = pad_and_merge(regions, margin_s=region_margin_s,
                                audio_duration=float(duration) if duration else None)
        say(f"Candidate regions after pad/merge: {len(regions)}")

        t0 = time.monotonic()
        try:
            wav_regions = extract_region_wavs(wav_path, regions, regions_dir)
        except LocalizationError:
            raise
        timings[f"region_extraction_t{tier_index}_s"] = round(
            time.monotonic() - t0, 3)

        t0 = time.monotonic()
        best = _validate_regions(wav_regions, target.strip(), fine_model,
                                 min_similarity, language, device, compute_type,
                                 log)
        timings[f"fine_asr_validation_t{tier_index}_s"] = round(
            time.monotonic() - t0, 3)

        if best is None:
            say(f"Tier {tier_name}: no region passed fine validation.")
            continue

        _, match, fine_result, winner_region, winner_wav = best

        # Forced alignment: refine ONLY the winning region's timestamps.
        onset_local = match.start_time
        aligned = False
        t0 = time.monotonic()
        try:
            refined = refine_word_timestamps(
                winner_wav, fine_result, language=language or "en", device=device)
            if refined is not None:
                refined_onset = match_refined_onset(match.matched_words, refined)
                if refined_onset is not None:
                    onset_local = refined_onset
                    aligned = True
        except Exception:
            pass  # Alignment must never fail the pipeline.
        timings["forced_alignment_s"] = round(time.monotonic() - t0, 3)
        if aligned:
            say(f"Forced alignment: applied (onset {onset_local:.3f}s local)")
        else:
            say("Forced alignment: skipped, using fine-ASR word onset "
                f"({onset_local:.3f}s local)")

        return CascadeResult(
            global_timestamp=winner_region.global_start + onset_local,
            matched_text=match.matched_text,
            similarity=match.text_similarity,
            average_word_probability=match.average_word_probability,
            minimum_word_probability=match.minimum_word_probability,
            fallback_tier=tier_name,
            coarse_model=coarse_model,
            fine_model=fine_model,
            aligned=aligned,
            region_count=len(regions),
            timings=timings,
            candidates_summary=[{
                "global_start": r.global_start,
                "global_end": r.global_end,
                "coarse_similarity": round(r.coarse_similarity, 4),
                "source_rank": r.source_rank,
            } for r in regions],
        )

    # Final tier: V1 behavior on the same WAV (never re-downloads/re-extracts).
    final_model = "small"
    say(f"All cascade tiers exhausted; falling back to full-audio "
        f"{final_model} ASR (V1 behavior).")
    t0 = time.monotonic()
    full = _transcribe(final_model, wav_path, language, device, compute_type)
    timings["full_asr_small_s"] = round(time.monotonic() - t0, 3)
    if not full.words:
        raise MatchNotFoundError(
            f"Target dialogue not found: final {final_model} transcript "
            f"contains no words.")
    matcher = DialogueMatcher(min_similarity=min_similarity)
    try:
        match = matcher.find_best_match_in_words(target.strip(), full.words)
    except MatcherError as e:
        raise MatchNotFoundError(f"Target dialogue not found: {e}") from e
    return CascadeResult(
        global_timestamp=match.start_time,
        matched_text=match.matched_text,
        similarity=match.text_similarity,
        average_word_probability=match.average_word_probability,
        minimum_word_probability=match.minimum_word_probability,
        fallback_tier="small-full",
        coarse_model=final_model,
        fine_model=final_model,
        aligned=False,
        region_count=0,
        timings=timings,
    )
