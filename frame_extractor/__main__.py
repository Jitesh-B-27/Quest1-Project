"""CLI for the V1 frame extractor.

Usage:
    python -m frame_extractor --video video/video.mp4 --timestamp 324.990
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from frame_extractor.core import FrameExtractor, FrameExtractorError

DEFAULT_OUTPUT = Path("frames")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m frame_extractor",
        description="Extract the video frame at a given timestamp.",
    )
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--timestamp", type=float, required=True,
                        help="Timestamp in seconds")
    parser.add_argument("--output", default=None,
                        help="Output image path or directory (default: frames/)")
    args = parser.parse_args(argv)

    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    # If it looks like an image path use it directly; otherwise treat as dir.
    if output.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        out_dir, out_name = output.parent, output.name
    else:
        out_dir, out_name = output, None

    try:
        extractor = FrameExtractor(output_dir=out_dir)
        result = extractor.extract_at_timestamp(
            args.video, args.timestamp, output_name=out_name,
        )
    except (FrameExtractorError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Video:                   {args.video}")
    print(f"Timestamp:               {result.timestamp:.3f}s")
    print(f"Frame number:            {result.frame_number}")
    print(f"Actual frame timestamp:  {result.actual_timestamp:.3f}s")
    print(f"FPS:                     {result.fps:.4f}")
    print(f"Resolution:              {result.width}x{result.height}")
    print(f"Duration:                {result.duration:.3f}s")
    print(f"Output image:            {result.image_path}")


if __name__ == "__main__":
    main()
