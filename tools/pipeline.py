"""
High-level pipeline orchestrator.

Each function is a resumable stage gate — it checks the state manager
before doing work and updates it after. The agent calls these functions
via tool use.
"""

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    DOWNLOAD_DIR,
    OUTPUT_DIR,
    REALESRGAN_MODEL,
    SAVED_PLAYLIST_ID,
    TARGET_CHANNEL_NAME,
    UPLOAD_PRIVACY,
)
from tools.downloader import download_video as _download
from tools.enhancer import encode_video, enhance_frames, extract_frames
from tools.state_manager import StateManager
from tools.youtube_tools import (
    get_video_details,
    list_playlist_videos,
    upload_video,
)

logger = logging.getLogger(__name__)
state = StateManager()


# ── 1. Discover saved videos ───────────────────────────────────────────────────

def fetch_saved_videos(playlist_id: str = SAVED_PLAYLIST_ID) -> Dict[str, Any]:
    """
    Pull the user's saved / Watch Later playlist from YouTube API.

    Returns a summary and registers any new videos as 'pending'.
    """
    try:
        videos = list_playlist_videos(playlist_id, max_results=500)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    # Enrich with duration info
    ids = [v["video_id"] for v in videos]
    details = {}
    try:
        details = get_video_details(ids)
    except Exception:
        pass

    new_count = 0
    for v in videos:
        vid = v["video_id"]
        det = details.get(vid, {})
        existing_stage = state.get_stage(vid)
        if existing_stage is None:
            state.upsert_video(
                vid,
                title=v["title"],
                channel=v["channel"],
                duration_secs=det.get("duration_secs"),
                stage="pending",
            )
            new_count += 1

    summary = state.summary()
    return {
        "success": True,
        "total_in_playlist": len(videos),
        "new_registered": new_count,
        "pipeline_summary": summary,
        "pending_videos": [
            {"video_id": v["video_id"], "title": v["title"]}
            for v in state.videos_by_stage("pending")
        ][:20],
    }


# ── 2. Download ────────────────────────────────────────────────────────────────

def download_video(video_id: str) -> Dict[str, Any]:
    """Download one video. Safe to call multiple times — idempotent."""
    v = state.get_video(video_id)
    if v is None:
        return {"success": False, "error": f"Video {video_id} not in state"}

    stage = v["stage"]
    if stage not in ("pending", "downloading", "failed"):
        return {
            "success": True,
            "skipped": True,
            "reason": f"Already at stage '{stage}'",
            "file_path": v.get("download_path"),
        }

    state.set_stage(video_id, "downloading")

    url = f"https://www.youtube.com/watch?v={video_id}"
    result = _download(url, video_id, DOWNLOAD_DIR)

    if not result["success"]:
        state.set_stage(video_id, "failed", error=result["error"])
        return {"success": False, "video_id": video_id, "error": result["error"]}

    state.upsert_video(
        video_id,
        stage="downloaded",
        download_path=result["file_path"],
        title=v.get("title") or result["title"],
    )
    return {
        "success": True,
        "video_id": video_id,
        "file_path": result["file_path"],
        "resolution": f"{result['width']}x{result['height']}",
        "fps": result["fps"],
        "duration_secs": result["duration"],
    }


# ── 3. Extract frames ──────────────────────────────────────────────────────────

def run_extract_frames(video_id: str) -> Dict[str, Any]:
    v = state.get_video(video_id)
    if v is None:
        return {"success": False, "error": "Unknown video_id"}

    stage = v["stage"]
    if stage not in ("downloaded", "extracting_frames", "failed"):
        return {
            "success": True,
            "skipped": True,
            "reason": f"Stage is '{stage}' — skipping extraction",
        }

    download_path = v.get("download_path")
    if not download_path or not Path(download_path).exists():
        state.set_stage(video_id, "failed", error="Download file missing")
        return {"success": False, "error": "Download file not found on disk"}

    state.set_stage(video_id, "extracting_frames")
    result = extract_frames(Path(download_path), video_id)

    if not result["success"]:
        state.set_stage(video_id, "failed", error=result.get("error"))
        return {"success": False, "error": result.get("error")}

    state.upsert_video(
        video_id,
        stage="frames_extracted",
        frame_count=result["frame_count"],
    )
    return {
        "success": True,
        "video_id": video_id,
        "frame_count": result["frame_count"],
        "fps": result["fps"],
        "frames_dir": result["frames_dir"],
    }


# ── 4. Enhance frames ──────────────────────────────────────────────────────────

def run_enhance_frames(
    video_id: str,
    model_name: str = REALESRGAN_MODEL,
) -> Dict[str, Any]:
    v = state.get_video(video_id)
    if v is None:
        return {"success": False, "error": "Unknown video_id"}

    stage = v["stage"]
    if stage not in ("frames_extracted", "enhancing", "failed"):
        return {
            "success": True,
            "skipped": True,
            "reason": f"Stage is '{stage}'",
        }

    state.set_stage(video_id, "enhancing")
    result = enhance_frames(video_id, model_name=model_name)

    if not result["success"]:
        state.set_stage(video_id, "failed", error=result.get("error"))
        return {"success": False, "error": result.get("error")}

    state.upsert_video(
        video_id,
        stage="enhanced",
        enhanced_frame_count=result["enhanced_count"],
    )
    return {
        "success": True,
        "video_id": video_id,
        "enhanced_count": result["enhanced_count"],
        "enhanced_dir": result["enhanced_dir"],
    }


# ── 5. Encode 4K video ─────────────────────────────────────────────────────────

def run_encode_video(video_id: str) -> Dict[str, Any]:
    v = state.get_video(video_id)
    if v is None:
        return {"success": False, "error": "Unknown video_id"}

    stage = v["stage"]
    if stage not in ("enhanced", "encoding", "failed"):
        return {
            "success": True,
            "skipped": True,
            "reason": f"Stage is '{stage}'",
        }

    download_path = v.get("download_path")
    if not download_path:
        state.set_stage(video_id, "failed", error="Download path missing")
        return {"success": False, "error": "No download path in state"}

    # Re-probe fps from download file
    from tools.enhancer import _get_fps
    fps = _get_fps(Path(download_path))

    state.set_stage(video_id, "encoding")
    result = encode_video(
        video_id=video_id,
        original_video_path=Path(download_path),
        fps=fps,
        title=v.get("title", video_id),
        output_dir=OUTPUT_DIR,
    )

    if not result["success"]:
        state.set_stage(video_id, "failed", error=result.get("error"))
        return {"success": False, "error": result.get("error")}

    state.upsert_video(
        video_id,
        stage="encoded",
        output_path=result["output_path"],
    )
    return {
        "success": True,
        "video_id": video_id,
        "output_path": result["output_path"],
    }


# ── 6. Upload to YouTube ───────────────────────────────────────────────────────

def run_upload_video(video_id: str) -> Dict[str, Any]:
    v = state.get_video(video_id)
    if v is None:
        return {"success": False, "error": "Unknown video_id"}

    stage = v["stage"]
    if stage not in ("encoded", "uploading", "failed"):
        return {
            "success": True,
            "skipped": True,
            "reason": f"Stage is '{stage}'",
        }

    output_path = v.get("output_path")
    if not output_path or not Path(output_path).exists():
        state.set_stage(video_id, "failed", error="Output file missing")
        return {"success": False, "error": "Encoded file not found"}

    original_title = v.get("title", "Restored Video")
    upload_title = f"[4K Restored] {original_title}"
    description = textwrap.dedent(f"""
        Originally uploaded by: {v.get('channel', 'Unknown')}
        Original video ID: {video_id}

        This video has been AI-upscaled to 4K using Real-ESRGAN.
        Denoised, enhanced, and upscaled from the original quality.

        Part of the "Down Memory Lane" restoration project —
        preserving and improving memories in stunning 4K clarity.
    """).strip()

    state.set_stage(video_id, "uploading")
    try:
        result = upload_video(
            file_path=Path(output_path),
            title=upload_title,
            description=description,
            tags=["4K", "restored", "AI upscaled", "Real-ESRGAN", "Down Memory Lane", "enhanced"],
            privacy_status=UPLOAD_PRIVACY,
        )
    except Exception as exc:
        state.set_stage(video_id, "failed", error=str(exc))
        return {"success": False, "error": str(exc)}

    state.upsert_video(
        video_id,
        stage="complete",
        uploaded_url=result["url"],
    )
    return {
        "success": True,
        "video_id": video_id,
        "uploaded_url": result["url"],
        "uploaded_title": result["title"],
    }


# ── Status helpers ─────────────────────────────────────────────────────────────

def get_pipeline_status() -> Dict[str, Any]:
    """Return overall pipeline progress."""
    return {
        "summary": state.summary(),
        "all_videos": state.all_videos(),
    }


def get_next_pending_video() -> Optional[Dict[str, Any]]:
    """Return the next video that needs work (not complete/skipped)."""
    for stage in ("pending", "downloaded", "frames_extracted", "enhanced", "encoded", "failed", "downloading", "extracting_frames", "enhancing", "encoding", "uploading"):
        videos = state.videos_by_stage(stage)
        if videos:
            return videos[0]
    return None
