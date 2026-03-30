"""
Video downloader using yt-dlp.

Downloads at the highest available resolution with audio merged.
Saves download cookies via browser extraction for age-gated content.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    DOWNLOAD_DIR,
    YTDLP_FORMAT,
    YTDLP_MERGE_FORMAT,
)

logger = logging.getLogger(__name__)


def download_video(
    video_url: str,
    video_id: str,
    output_dir: Path = DOWNLOAD_DIR,
) -> Dict[str, Any]:
    """
    Download a YouTube video at highest quality.

    Returns:
        {
            "success": bool,
            "file_path": str | None,
            "title": str | None,
            "width": int | None,
            "height": int | None,
            "fps": float | None,
            "duration": float | None,
            "error": str | None,
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output template — keep video_id in filename for uniqueness
    output_template = str(output_dir / f"{video_id}_%(title)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", YTDLP_FORMAT,
        "--merge-output-format", YTDLP_MERGE_FORMAT,
        "--output", output_template,
        "--write-info-json",             # save metadata
        "--no-playlist",                 # don't expand playlists
        "--continue",                    # resume partial downloads
        "--retries", "5",
        "--fragment-retries", "5",
        "--socket-timeout", "30",
        "--no-warnings",
        "--progress",
        video_url,
    ]

    logger.info("Downloading %s …", video_url)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        logger.error("yt-dlp failed: %s", error_msg)
        return {
            "success": False,
            "file_path": None,
            "title": None,
            "width": None,
            "height": None,
            "fps": None,
            "duration": None,
            "error": error_msg[:500],
        }

    # Find the downloaded file
    mp4_files = sorted(
        output_dir.glob(f"{video_id}_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not mp4_files:
        return {
            "success": False,
            "file_path": None,
            "title": None,
            "width": None,
            "height": None,
            "fps": None,
            "duration": None,
            "error": "Downloaded file not found on disk",
        }

    file_path = mp4_files[0]
    meta = _probe_video(file_path)
    logger.info("Downloaded → %s (%dx%d @ %.2f fps)", file_path.name, meta["width"], meta["height"], meta["fps"])

    return {
        "success": True,
        "file_path": str(file_path),
        "title": meta.get("title") or file_path.stem.replace(f"{video_id}_", ""),
        "width": meta["width"],
        "height": meta["height"],
        "fps": meta["fps"],
        "duration": meta["duration"],
        "error": None,
    }


def _probe_video(file_path: Path) -> Dict[str, Any]:
    """Use ffprobe to get video stream metadata."""
    import json
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(file_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out)
        stream = data.get("streams", [{}])[0]
        # Parse fps fraction like "30000/1001"
        fps_str = stream.get("r_frame_rate", "0/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": round(fps, 3),
            "duration": float(stream.get("duration", 0)),
            "title": None,
        }
    except Exception as exc:
        logger.warning("ffprobe failed: %s", exc)
        return {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0, "title": None}
