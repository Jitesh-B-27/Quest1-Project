import argparse
import shutil
import sys
import time
from pathlib import Path

import yt_dlp

VIDEO_DIR = Path("video")
LOCAL_FFMPEG = Path(__file__).resolve().parent / "tools" / "ffmpeg" / "bin"

DEFAULT_RETRIES = 5
RETRY_DELAY_SECONDS = 10


def find_ffmpeg() -> str | None:
    if LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)
    return shutil.which("ffmpeg")


def build_ydl_opts(output_dir: Path, proxy: str | None = None,
                   retries: int = DEFAULT_RETRIES) -> dict:
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
    if proxy:
        opts["proxy"] = proxy
    opts.update({
        "outtmpl": str(output_dir / "%(title).100s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "nocheckcertificate": True,
        # Retry transient network errors at the HTTP level as well.
        "retries": retries,
        "fragment_retries": retries,
    })
    return opts


def download_video(
    url: str,
    output_dir: Path = VIDEO_DIR,
    proxy: str | None = None,
    attempts: int = DEFAULT_RETRIES,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> Path:
    """Download a video and return the local path to the resulting MP4 file.

    Retries on transient network/SSL errors with a delay between attempts,
    since some networks intermittently reset connections to certain hosts.
    """
    if not url or not url.strip():
        raise yt_dlp.utils.DownloadError("URL cannot be empty.")

    # Console may use a legacy codepage (cp1252); never crash on unicode titles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = build_ydl_opts(output_dir, proxy)
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise yt_dlp.utils.DownloadError("Could not extract video info.")
                title = info.get("title", "video")
                filename = Path(ydl.prepare_filename(info))
                # merge_output_format guarantees mp4; fix extension if it differs.
                if filename.suffix.lower() != ".mp4":
                    filename = filename.with_suffix(".mp4")
                print(f"Downloaded: {title}")
                print(f"Saved to: {filename}")
                return filename
        except KeyboardInterrupt:
            raise
        except (yt_dlp.utils.DownloadError,) as e:
            last_error = e
            if attempt < attempts:
                print(
                    f"Attempt {attempt}/{attempts} failed ({type(e).__name__}); "
                    f"retrying in {retry_delay}s...",
                    file=sys.stderr,
                )
                time.sleep(retry_delay)

    raise last_error  # type: ignore[misc]


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
    parser.add_argument(
        "--proxy",
        default=None,
        help="Proxy URL for downloading, e.g. socks5://host:port or http://host:port",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_RETRIES,
        help="Number of download attempts on transient network errors "
             "(default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        download_video(args.url, args.output_dir, args.proxy, args.attempts)
        sys.exit(0)
    except yt_dlp.utils.DownloadError as e:
        print(f"Download failed after {args.attempts} attempts: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
