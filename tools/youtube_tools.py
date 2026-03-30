"""
YouTube Data API v3 helpers.

Handles:
  - OAuth2 authentication (first run: browser flow; subsequent: token refresh)
  - Listing videos from a playlist (Saved / Watch Later / custom)
  - Uploading a video to the target channel
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Google API client
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from config import (
    TARGET_CHANNEL_NAME,
    UPLOAD_PRIVACY,
    YOUTUBE_CLIENT_SECRETS,
    YOUTUBE_TOKEN_FILE,
)

logger = logging.getLogger(__name__)

# Scopes required: read playlists + upload videos
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


# ── Auth ───────────────────────────────────────────────────────────────────────

def _get_credentials() -> Credentials:
    """Load or refresh OAuth2 credentials, running the browser flow if needed."""
    creds: Optional[Credentials] = None

    if os.path.exists(YOUTUBE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(YOUTUBE_TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(YOUTUBE_CLIENT_SECRETS):
                raise FileNotFoundError(
                    f"Missing OAuth2 client secrets: {YOUTUBE_CLIENT_SECRETS}\n"
                    "Download from Google Cloud Console → APIs & Services → Credentials"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_CLIENT_SECRETS, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(YOUTUBE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        logger.info("YouTube credentials saved to %s", YOUTUBE_TOKEN_FILE)

    return creds


def get_youtube_client():
    """Return an authenticated YouTube API client."""
    creds = _get_credentials()
    return build("youtube", "v3", credentials=creds)


# ── Playlist / Saved videos ────────────────────────────────────────────────────

def list_playlist_videos(playlist_id: str, max_results: int = 200) -> List[Dict[str, Any]]:
    """
    Return a list of video dicts from a playlist.

    Each dict contains:
      video_id, title, channel, published_at, thumbnail_url, duration
    """
    youtube = get_youtube_client()
    videos: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page_token,
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]
            video_id = item["contentDetails"]["videoId"]
            videos.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", "Untitled"),
                    "channel": snippet.get("videoOwnerChannelTitle", "Unknown"),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail_url": (
                        snippet.get("thumbnails", {})
                        .get("maxres", snippet.get("thumbnails", {}).get("high", {}))
                        .get("url", "")
                    ),
                    "playlist_position": snippet.get("position", 0),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(videos) >= max_results:
            break

    logger.info("Found %d videos in playlist %s", len(videos), playlist_id)
    return videos


def get_video_details(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch duration and other details for a list of video IDs."""
    youtube = get_youtube_client()
    details: Dict[str, Dict[str, Any]] = {}

    # API allows max 50 IDs per request
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        response = youtube.videos().list(
            part="contentDetails,statistics",
            id=",".join(chunk),
        ).execute()

        for item in response.get("items", []):
            vid = item["id"]
            duration_iso = item["contentDetails"].get("duration", "PT0S")
            details[vid] = {
                "duration_iso": duration_iso,
                "duration_secs": _parse_duration(duration_iso),
                "view_count": item.get("statistics", {}).get("viewCount", 0),
                "definition": item["contentDetails"].get("definition", "sd"),
            }

    return details


def _parse_duration(iso: str) -> int:
    """Convert ISO 8601 duration (PT4M13S) to seconds."""
    import re
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso
    )
    if not match:
        return 0
    h, m, s = (int(x or 0) for x in match.groups())
    return h * 3600 + m * 60 + s


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_video(
    file_path: Path,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    category_id: str = "22",   # 22 = People & Blogs; 28 = Science & Technology
    privacy_status: str = UPLOAD_PRIVACY,
) -> Dict[str, Any]:
    """
    Upload a video file to the authenticated user's YouTube channel.

    Returns a dict with 'video_id' and 'url'.
    Uses resumable upload so large files don't fail on network hiccups.
    """
    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": title[:100],  # YouTube max title length
            "description": description[:5000],
            "tags": tags or ["4K", "restored", "enhanced", "upscaled"],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(file_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    retry = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                logger.info("Upload progress: %d%%", pct)
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retry < 5:
                retry += 1
                wait = 2 ** retry
                logger.warning("HTTP %d — retry %d in %ds", e.resp.status, retry, wait)
                time.sleep(wait)
            else:
                raise

    video_id = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info("Uploaded! %s → %s", title, url)
    return {"video_id": video_id, "url": url, "title": title}


# ── Channel discovery ──────────────────────────────────────────────────────────

def get_my_channel_info() -> Dict[str, Any]:
    """Return the authenticated user's channel details."""
    youtube = get_youtube_client()
    response = youtube.channels().list(part="snippet,contentDetails", mine=True).execute()
    items = response.get("items", [])
    if not items:
        return {}
    ch = items[0]
    return {
        "channel_id": ch["id"],
        "title": ch["snippet"]["title"],
        "uploads_playlist": ch["contentDetails"]["relatedPlaylists"]["uploads"],
    }
