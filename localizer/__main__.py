"""CLI for the V2 coarse-to-fine localization cascade.

Usage:
    python -m localizer --wav audio/full.wav --target "My mind rebels at stagnation"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asr.aligner import AlignmentError  # noqa: F401  (exposed for users)
from localizer.cascade import CascadeResult, run_cascade
from localizer.core import (
    REGIONS_DIR,
    LocalizationError,
    cleanup_regions,
    format_timestamp,
)
from matcher.core import MatchNotFoundError, MatcherError


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m localizer",
        description="Locate target dialogue via coarse-to-fine cascade "
                    "(tiny/base full-audio -> small region ASR -> alignment).",
    )
    parser.add_argument("--wav", type=Path, default=Path("audio/audio.wav"),
                        help="Full-audio WAV file (default: %(default)s)")
    parser.add_argument("--target", required=True,
                        help="Target dialogue sentence to locate")
    parser.add_argument("--coarse-tiers", default="tiny,base",
                        help="Comma-separated coarse models tried in order "
                             "(default: %(default)s)")
    parser.add_argument("--fine-model", default="small",
                        help="Fine ASR model for candidate regions "
                             "(default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of candidate regions (default: %(default)s)")
    parser.add_argument("--region-margin", type=float, default=5.0,
                        help="Padding seconds around each region "
                             "(default: %(default)s)")
    parser.add_argument("--min-similarity", type=float, default=0.7,
                        help="Fine-stage validation gate (default: %(default)s)")
    parser.add_argument("--language", default="en",
                        help="Language code; 'none' for auto-detect "
                             "(default: %(default)s)")
    parser.add_argument("--regions-dir", type=Path, default=REGIONS_DIR,
                        help="Directory for temp region WAVs "
                             "(default: %(default)s)")
    parser.add_argument("--keep-regions", action="store_true",
                        help="Keep temp region WAVs after success")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the result JSON")
    args = parser.parse_args(argv)

    tiers = tuple(t.strip() for t in args.coarse_tiers.split(",") if t.strip())
    language = None if args.language.lower() == "none" else args.language

    try:
        result = run_cascade(
            wav_path=args.wav,
            target=args.target,
            coarse_tiers=tiers,
            fine_model=args.fine_model,
            top_k=args.top_k,
            region_margin_s=args.region_margin,
            min_similarity=args.min_similarity,
            language=language,
            regions_dir=args.regions_dir,
            log=lambda msg: print(msg),
        )
        if not args.keep_regions:
            cleanup_regions(args.regions_dir)
    except (MatchNotFoundError, LocalizationError, MatcherError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    payload = _to_payload(args.target, result)
    print("\nMatch found.")
    print(f"  Text:          {result.matched_text}")
    print(f"  Timestamp:     {result.global_timestamp:.3f}s  "
          f"({format_timestamp(result.global_timestamp)})")
    print(f"  Similarity:    {result.similarity:.4f}")
    print(f"  Tier:          {result.fallback_tier}  Aligned: {result.aligned}")

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2),
                                   encoding="utf-8")
            print(f"  Result JSON:   {args.output}")
        except OSError as e:
            print(f"Error: could not write output file: {e}", file=sys.stderr)
            sys.exit(1)


def _to_payload(target: str, r: CascadeResult) -> dict:
    return {
        "target_text": target,
        "matched_text": r.matched_text,
        "global_timestamp_seconds": r.global_timestamp,
        "timestamp_hhmmss": format_timestamp(r.global_timestamp),
        "similarity": round(r.similarity, 4),
        "average_word_probability": round(r.average_word_probability, 4),
        "minimum_word_probability": round(r.minimum_word_probability, 4),
        "aligned": r.aligned,
        "fallback_tier": r.fallback_tier,
        "coarse_model": r.coarse_model,
        "fine_model": r.fine_model,
        "region_count": r.region_count,
        "timings": r.timings,
    }


# Local alias removed; MatcherError imported above.


if __name__ == "__main__":
    main()
