"""CLI for the V1 dialogue matcher.

Usage:
    python -m matcher --transcript transcript/transcript.json \
        --target "My mind rebels at stagnation"
"""

from __future__ import annotations

import argparse
import json
import sys

from matcher.core import (
    DialogueMatcher,
    MatcherError,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m matcher",
        description="Find the best match for a target dialogue in a transcript.",
    )
    parser.add_argument("--transcript", default="transcript/transcript.json",
                        help="Path to transcript.json (default: %(default)s)")
    parser.add_argument("--target", required=True,
                        help="Target dialogue text to find")
    parser.add_argument("--min-similarity", type=float, default=0.7,
                        help="Minimum similarity for a valid match (default: %(default)s)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top candidates to retain (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="Optional path to write the full result as JSON")
    args = parser.parse_args(argv)

    try:
        matcher = DialogueMatcher(top_n=args.top_n, min_similarity=args.min_similarity)
        result = matcher.find_best_match(args.target, args.transcript)
    except MatcherError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Target:                {result.target_text}")
    print(f"Best matched text:     {result.matched_text}")
    print(f"Start timestamp:       {result.start_time:.3f}")
    print(f"End timestamp:         {result.end_time:.3f}")
    print(f"Similarity:            {result.text_similarity:.4f}")
    print(f"Average probability:   {result.average_word_probability:.4f}")
    print(f"Minimum probability:   {result.minimum_word_probability:.4f}")

    print("\nTop candidates:")
    for i, c in enumerate(result.candidates, 1):
        print(
            f"  {i}. sim={c.text_similarity:.4f} "
            f"avg_p={c.average_word_probability:.4f} "
            f"[{c.start_time:.2f}s - {c.end_time:.2f}s] '{c.matched_text}'"
        )

    if args.output:
        try:
            out = __import__("pathlib").Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            print(f"\nResult written to: {out}")
        except OSError as e:
            print(f"Error: could not write output file: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
