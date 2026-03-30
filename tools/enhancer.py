"""
AI-powered video enhancement pipeline.

Stage 1 — Frame extraction (ffmpeg)
Stage 2 — Per-frame upscaling with Real-ESRGAN (RTX 4070 accelerated)
Stage 3 — 4K video reconstruction with original audio

Models available (all run on RTX 4070 / 12 GB VRAM):
  realesr-generalvideoV3   — temporal-stable, best for video (DEFAULT)
  RealESRGAN_x4plus        — highest quality for general stills/video
  RealESRGAN_x4plus_anime_6B — anime/cartoon content

Uses tiled processing so VRAM usage stays bounded regardless of input size.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from config import (
    CLEANUP_FRAMES,
    DENOISE_STRENGTH,
    ENHANCED_DIR,
    FFMPEG_THREADS,
    FRAMES_DIR,
    OUTPUT_DIR,
    REALESRGAN_FP32,
    REALESRGAN_MODEL,
    REALESRGAN_SCALE,
    REALESRGAN_TILE,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

logger = logging.getLogger(__name__)


# ── Model factory ──────────────────────────────────────────────────────────────

def _build_upsampler(model_name: str = REALESRGAN_MODEL) -> RealESRGANer:
    """
    Instantiate the Real-ESRGAN upsampler for the chosen model.
    Downloads weights automatically on first run.
    """
    half = not REALESRGAN_FP32  # FP16 on RTX 40xx is safe and fast

    if model_name == "realesr-generalvideoV3":
        # SRVGGNet — designed for video, temporally smooth
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_conv=16,
            upscale=4, act_type="prelu",
        )
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.5.0/realesr-generalvideoV3.pth"
        )
        netscale = 4

    elif model_name == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=4,
        )
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        )
        netscale = 4

    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=6,
            num_grow_ch=32, scale=4,
        )
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        )
        netscale = 4

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Store weights in project directory
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_url,
        model=model,
        tile=REALESRGAN_TILE,
        tile_pad=10,
        pre_pad=0,
        half=half,
        gpu_id=0,
    )
    logger.info("Loaded model: %s (tile=%d, fp16=%s)", model_name, REALESRGAN_TILE, half)
    return upsampler


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(
    video_path: Path,
    video_id: str,
) -> Dict[str, Any]:
    """
    Extract all frames from the input video using ffmpeg.

    Returns:
        {"success": bool, "frames_dir": str, "frame_count": int, "fps": float}
    """
    frames_dir = FRAMES_DIR / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing = sorted(frames_dir.glob("frame_*.png"))
    if existing:
        logger.info("Frames already extracted: %d frames in %s", len(existing), frames_dir)
        fps = _get_fps(video_path)
        return {
            "success": True,
            "frames_dir": str(frames_dir),
            "frame_count": len(existing),
            "fps": fps,
        }

    fps = _get_fps(video_path)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", "1",                  # highest quality PNG
        "-threads", str(FFMPEG_THREADS),
        str(frames_dir / "frame_%06d.png"),
        "-y",
    ]
    logger.info("Extracting frames from %s …", video_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {
            "success": False,
            "frames_dir": str(frames_dir),
            "frame_count": 0,
            "fps": fps,
            "error": result.stderr[-500:],
        }

    frame_count = len(sorted(frames_dir.glob("frame_*.png")))
    logger.info("Extracted %d frames (%.3f fps)", frame_count, fps)
    return {
        "success": True,
        "frames_dir": str(frames_dir),
        "frame_count": frame_count,
        "fps": fps,
    }


def _get_fps(video_path: Path) -> float:
    import json
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        stream = json.loads(out).get("streams", [{}])[0]
        num, den = stream.get("r_frame_rate", "30/1").split("/")
        return round(float(num) / float(den), 3)
    except Exception:
        return 30.0


# ── Enhancement ────────────────────────────────────────────────────────────────

def enhance_frames(
    video_id: str,
    frames_dir: Optional[Path] = None,
    model_name: str = REALESRGAN_MODEL,
) -> Dict[str, Any]:
    """
    Upscale all extracted frames with Real-ESRGAN.

    Resumes automatically — already-enhanced frames are skipped.

    Returns:
        {"success": bool, "enhanced_dir": str, "enhanced_count": int}
    """
    if frames_dir is None:
        frames_dir = FRAMES_DIR / video_id

    enhanced_dir = ENHANCED_DIR / video_id
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(frames_dir.glob("frame_*.png"))
    if not all_frames:
        return {
            "success": False,
            "enhanced_dir": str(enhanced_dir),
            "enhanced_count": 0,
            "error": "No frames found to enhance",
        }

    # Determine which frames still need processing
    todo = [
        f for f in all_frames
        if not (enhanced_dir / f.name).exists()
    ]

    logger.info(
        "Enhancing %d/%d frames (skipping %d already done) …",
        len(todo), len(all_frames), len(all_frames) - len(todo),
    )

    if not todo:
        return {
            "success": True,
            "enhanced_dir": str(enhanced_dir),
            "enhanced_count": len(all_frames),
        }

    upsampler = _build_upsampler(model_name)

    errors = 0
    for i, frame_path in enumerate(todo):
        try:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Could not read frame: %s", frame_path.name)
                errors += 1
                continue

            output, _ = upsampler.enhance(img, outscale=REALESRGAN_SCALE)
            out_path = enhanced_dir / frame_path.name
            cv2.imwrite(str(out_path), output)

            if (i + 1) % 100 == 0:
                logger.info("  %d / %d frames enhanced …", i + 1, len(todo))

        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                logger.error("OOM — reduce REALESRGAN_TILE in config.py and retry")
                raise
            logger.warning("Frame %s failed: %s", frame_path.name, exc)
            errors += 1

    enhanced_count = len(sorted(enhanced_dir.glob("frame_*.png")))
    logger.info("Enhancement done: %d frames, %d errors", enhanced_count, errors)

    return {
        "success": errors == 0,
        "enhanced_dir": str(enhanced_dir),
        "enhanced_count": enhanced_count,
        "error_count": errors,
    }


# ── Video reconstruction ───────────────────────────────────────────────────────

def encode_video(
    video_id: str,
    original_video_path: Path,
    fps: float,
    title: str = "",
    enhanced_dir: Optional[Path] = None,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Reconstruct the enhanced 4K video from upscaled frames + original audio.

    Uses H.265/HEVC for high-quality compression at 4K.

    Returns:
        {"success": bool, "output_path": str}
    """
    if enhanced_dir is None:
        enhanced_dir = ENHANCED_DIR / video_id

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
    output_path = output_dir / f"{video_id}_{safe_title}_4K.mp4"

    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        logger.info("Output already exists: %s", output_path.name)
        return {"success": True, "output_path": str(output_path)}

    frames = sorted(enhanced_dir.glob("frame_*.png"))
    if not frames:
        return {
            "success": False,
            "output_path": None,
            "error": "No enhanced frames found",
        }

    logger.info("Encoding 4K video from %d frames at %.3f fps …", len(frames), fps)

    # Two-pass encode: frames → H.265 + mux original audio
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(enhanced_dir / "frame_%06d.png"),  # upscaled frames
        "-i", str(original_video_path),               # original (for audio)
        "-map", "0:v:0",    # video from enhanced frames
        "-map", "1:a?",     # audio from original (optional)
        "-c:v", "libx265",
        "-preset", "medium",
        "-crf", "18",       # high quality: lower = better, 18 is near-lossless
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-movflags", "+faststart",
        "-threads", str(FFMPEG_THREADS),
        str(output_path),
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("FFmpeg encode failed:\n%s", result.stderr[-1000:])
        return {
            "success": False,
            "output_path": None,
            "error": result.stderr[-500:],
        }

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("Encoded → %s (%.1f MB)", output_path.name, size_mb)

    # Optionally clean up raw and enhanced frames to save disk
    if CLEANUP_FRAMES:
        raw_dir = FRAMES_DIR / video_id
        enh_dir = ENHANCED_DIR / video_id
        for d in (raw_dir, enh_dir):
            if d.exists():
                shutil.rmtree(d)
                logger.info("Cleaned up frames: %s", d)

    return {"success": True, "output_path": str(output_path)}
