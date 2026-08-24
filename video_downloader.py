import argparse
import shutil
import sys
from pathlib import Path

import yt_dlp

VIDEO_DIR = Path("video")
LOCAL_FFMPEG = Path(__file__).resolve().parent / "tools" / "ffmpeg" / "bin"


def find_ffmpeg() -> str | None:
    if LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)
    return shutil.which("ffmpeg")


def build_ydl_opts(output_dir: Path) -> dict:
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_path,
        }
    else:
        print(
            "Warning: ffmpeg not found; falling back to best single-file format "
            "(lower quality than merged video+audio).",
            file=sys.stderr,
        )
        opts = {
            "format": "best[ext=mp4]/best",
        }
    opts.update({
        "outtmpl": str(output_dir / "%(title).100s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "nocheckcertificate": True,
    })
    return opts


def download_video(url: str, output_dir: Path = VIDEO_DIR) -> int:
    if not url or not url.strip():
        print("Error: URL cannot be empty.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = build_ydl_opts(output_dir)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                print("Error: Could not extract video info.", file=sys.stderr)
                return 1
            title = info.get("title", "video")
            filename = ydl.prepare_filename(info)
            print(f"Downloaded: {title}")
            print(f"Saved to: {filename}")
            return 0
    except yt_dlp.utils.DownloadError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.", file=sys.stderr)
        return 130


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a video (YouTube, Vimeo, etc.) and save it locally as MP4."
    )
    parser.add_argument("url", help="URL of the video to download")
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=VIDEO_DIR,
        help="Directory to save the video (default: ./video)",
    )
    args = parser.parse_args()

    sys.exit(download_video(args.url, args.output_dir))


if __name__ == "__main__":
    main()
