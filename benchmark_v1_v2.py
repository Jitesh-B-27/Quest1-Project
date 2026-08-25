"""Benchmark V1 vs V2 pipelines on the production video.

Prerequisite: media cache exists (video/ and audio/ populated), e.g. after
one full download run. This script never downloads or re-extracts media.

Usage:
    python benchmark_v1_v2.py --target "My mind rebels at stagnation"
    python benchmark_v1_v2.py --target "..." --repeats 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pipeline
from localizer.core import format_timestamp


def _run_v1(video: Path, wav: Path, target: str, model_size: str):
    t0 = time.monotonic()
    result = pipeline.run_pipeline(
        video_path=str(video),
        audio_path=str(wav),
        target=target,
        model_size=model_size,
        model_type="faster-whisper",
        save_result=False,
    )
    return {
        "arch": "v1",
        "models": model_size,
        "total_s": round(time.monotonic() - t0, 1),
        "timestamp": result.timestamp,
        "hhmmss": format_timestamp(result.timestamp),
        "frame": result.frame_number,
        "similarity": result.similarity,
    }


def _run_v2(video: Path, wav: Path, target: str,
            coarse_tiers: tuple[str, ...], fine_model: str,
            top_k: int, margin: float):
    t0 = time.monotonic()
    result = pipeline.run_pipeline_v2(
        video_path=str(video),
        audio_path=str(wav),
        target=target,
        coarse_tiers=coarse_tiers,
        fine_model=fine_model,
        top_k=top_k,
        region_margin_s=margin,
        save_result=False,
    )
    return {
        "arch": "v2",
        "models": f"{'/'.join(coarse_tiers)}->{fine_model}",
        "total_s": round(time.monotonic() - t0, 1),
        "timestamp": result.timestamp,
        "hhmmss": result.timestamp_hhmmss,
        "frame": result.frame_number,
        "similarity": result.similarity,
        "tier": result.fallback_tier,
        "aligned": result.aligned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V1 vs V2 on the production video.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--v1-model", default="small")
    parser.add_argument("--coarse-tiers", default="tiny,base")
    parser.add_argument("--fine-model", default="small")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--region-margin", type=float, default=5.0)
    args = parser.parse_args()

    video = pipeline.resolve_media_path("video", pipeline.VIDEO_EXTENSIONS, "video")
    wav = pipeline.resolve_media_path("audio", pipeline.AUDIO_EXTENSIONS, "audio")

    rows = [
        _run_v1(video, wav, args.target, args.v1_model),
        _run_v2(video, wav, args.target,
                tuple(t.strip() for t in args.coarse_tiers.split(",")),
                args.fine_model, args.top_k, args.region_margin),
    ]

    v1, v2 = rows[0], rows[1]
    delta_ts = abs(v2["timestamp"] - v1["timestamp"])

    print("\n" + "=" * 78)
    print(f"BENCHMARK  target='{args.target}'")
    print("=" * 78)
    header = f"{'arch':<5} {'models':<14} {'time_s':>8} {'timestamp':>14} {'frame':>7} {'sim':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        extra = f"  tier={r.get('tier', '')} aligned={r.get('aligned', '')}"
        print(f"{r['arch']:<5} {r['models']:<14} {r['total_s']:>8} "
              f"{r['hhmmss']:>14} {r['frame']:>7} {r['similarity']:>6.4f}{extra}")
    print("-" * len(header))
    if v1["total_s"] > 0:
        print(f"Speedup: {v1['total_s'] / v2['total_s']:.2f}x   "
              f"|timestamp delta|: {delta_ts:.3f}s")

    out = Path("output") / "benchmark_v1_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
