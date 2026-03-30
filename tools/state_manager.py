"""
Persistent state manager — tracks every video through the pipeline.
Survives crashes and restarts so the agent can resume mid-job.

State machine per video:
  pending → downloading → downloaded → extracting_frames → frames_extracted
  → enhancing → enhanced → encoding → encoded → uploading → complete
  (any stage → failed)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import STATE_FILE

logger = logging.getLogger(__name__)

# Valid pipeline stages
STAGES = [
    "pending",
    "downloading",
    "downloaded",
    "extracting_frames",
    "frames_extracted",
    "enhancing",
    "enhanced",
    "encoding",
    "encoded",
    "uploading",
    "complete",
    "failed",
    "skipped",
]


class StateManager:
    """JSON-backed state store with atomic writes."""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self._data: Dict[str, Any] = self._load()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupt state file — starting fresh.")
        return {"videos": {}, "last_updated": None}

    def _save(self) -> None:
        self._data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self.state_file)

    # ── Video records ─────────────────────────────────────────────────────────

    def upsert_video(self, video_id: str, **fields) -> None:
        """Create or update a video record."""
        if video_id not in self._data["videos"]:
            self._data["videos"][video_id] = {
                "video_id": video_id,
                "stage": "pending",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "updated_at": None,
                "error": None,
                "download_path": None,
                "output_path": None,
                "uploaded_url": None,
                "title": None,
                "channel": None,
                "duration_secs": None,
                "frame_count": None,
                "enhanced_frame_count": None,
            }
        self._data["videos"][video_id].update(fields)
        self._data["videos"][video_id]["updated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        self._save()

    def set_stage(self, video_id: str, stage: str, error: Optional[str] = None) -> None:
        assert stage in STAGES, f"Unknown stage: {stage}"
        self.upsert_video(video_id, stage=stage, error=error)
        logger.info("[%s] → %s%s", video_id, stage, f" ({error})" if error else "")

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        return self._data["videos"].get(video_id)

    def get_stage(self, video_id: str) -> Optional[str]:
        v = self.get_video(video_id)
        return v["stage"] if v else None

    def all_videos(self) -> List[Dict[str, Any]]:
        return list(self._data["videos"].values())

    def videos_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        return [v for v in self.all_videos() if v["stage"] == stage]

    def pending_or_failed(self) -> List[Dict[str, Any]]:
        """Videos that still need processing (including failed retries)."""
        return [
            v for v in self.all_videos()
            if v["stage"] not in ("complete", "skipped")
        ]

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s: 0 for s in STAGES}
        for v in self.all_videos():
            counts[v["stage"]] = counts.get(v["stage"], 0) + 1
        return {k: v for k, v in counts.items() if v > 0}

    def __repr__(self) -> str:
        return f"StateManager({self.summary()})"
