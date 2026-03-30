"""
Configuration for YouTube 4K Enhancer Agent.
All secrets come from environment variables — never hardcoded.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DOWNLOAD_DIR = BASE_DIR / "downloads"
FRAMES_DIR   = BASE_DIR / "frames"
ENHANCED_DIR = BASE_DIR / "enhanced"
OUTPUT_DIR   = BASE_DIR / "output"
STATE_FILE   = BASE_DIR / "state.json"
LOG_FILE     = BASE_DIR / "agent.log"

for d in (DOWNLOAD_DIR, FRAMES_DIR, ENHANCED_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys (from .env) ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
YOUTUBE_API_KEY     = os.environ.get("YOUTUBE_API_KEY", "")

# OAuth2 credentials JSON file path (downloaded from Google Cloud Console)
YOUTUBE_CLIENT_SECRETS = os.environ.get(
    "YOUTUBE_CLIENT_SECRETS",
    str(BASE_DIR / "client_secrets.json"),
)
# Token file persisted after first OAuth dance
YOUTUBE_TOKEN_FILE = str(BASE_DIR / "youtube_token.json")

# ── Claude Model ───────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-opus-4-6"

# ── YouTube ────────────────────────────────────────────────────────────────────
# The saved-videos playlist ID for the authenticated account
# "Saved" playlist uses "WL" (Watch Later); "Liked" uses "LL"
# For the user's actual "Saved" playlist use the API to discover it.
# We default to Watch Later — change via env if needed.
SAVED_PLAYLIST_ID = os.environ.get("SAVED_PLAYLIST_ID", "WL")

TARGET_CHANNEL_NAME  = "Down Memory Lane"
TARGET_CHANNEL_HANDLE = "downmemorylane"

# ── yt-dlp ────────────────────────────────────────────────────────────────────
YTDLP_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"
YTDLP_MERGE_FORMAT = "mp4"

# ── Real-ESRGAN ────────────────────────────────────────────────────────────────
# Model choices:
#   "RealESRGAN_x4plus"           – general purpose, great quality
#   "realesr-generalvideoV3"      – video-optimised, faster
#   "RealESRGAN_x4plus_anime_6B"  – anime/cartoon content
REALESRGAN_MODEL   = "realesr-generalvideoV3"
REALESRGAN_SCALE   = 4          # target 4× upscale → SD/HD → 4K
REALESRGAN_TILE    = 512        # tile size for RTX 4070 (12 GB VRAM)
REALESRGAN_FP32    = False      # use FP16 for speed on RTX 4070
DENOISE_STRENGTH   = 0.5        # 0–1, higher = more denoising

# FFmpeg thread count (leave headroom for GPU)
FFMPEG_THREADS = 4

# ── Processing ────────────────────────────────────────────────────────────────
# Max videos to process per agent run (set None for unlimited)
MAX_VIDEOS_PER_RUN = None

# Whether to delete intermediate frames after encoding (saves disk)
CLEANUP_FRAMES = True

# Target output resolution (sanity-check after upscale)
TARGET_WIDTH  = 3840
TARGET_HEIGHT = 2160

# Upload privacy: "public", "unlisted", "private"
UPLOAD_PRIVACY = "private"   # start private so you can review before publishing
