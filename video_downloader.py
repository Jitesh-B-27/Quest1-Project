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


def download_video(url: str, output_dir: Path = VIDEO_DIR) -> Path:
    """Download a video and return the local path to the resulting MP4 file."""
    if not url or not url.strip():
        raise yt_dlp.utils.DownloadError("URL cannot be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = build_ydl_opts(output_dir)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise yt_dlp.utils.DownloadError("Could not extract video info.")
        title = info.get("title", "video")
        filename = Path(ydl.prepare_filename(info))
        # merge_output_format guarantees mp4; fix extension if it differs.
        if filename.suffix.lower() != ".mp4":
            mp4 = filename.with_suffix(".mp4")
            if mp4.exists():
                filename = mp4
            else:
                filename = filename.with_suffix(".mp4")
        print(f"Downloaded: {title}")
        print(f"Saved to: {filename}")
        return filename


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

    try:
        download_video(args.url, args.output_dir)
        sys.exit(0)
    except yt_dlp.utils.DownloadError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
