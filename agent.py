"""
YouTube 4K Enhancer — Claude Opus Agent

An agentic pipeline orchestrated by Claude (claude-opus-4-6) that:
  1. Fetches videos from the user's YouTube Saved/Watch-Later playlist
  2. Downloads each at highest resolution
  3. AI-upscales every frame to 4K with Real-ESRGAN (RTX 4070 accelerated)
  4. Reconstructs the 4K video with original audio
  5. Uploads to the "Down Memory Lane" YouTube channel
  6. Resumes seamlessly after any crash

Usage:
    python agent.py                  # full run
    python agent.py --status         # show pipeline status only
    python agent.py --video VIDEO_ID # process one specific video
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, REALESRGAN_MODEL, SAVED_PLAYLIST_ID
from tools.pipeline import (
    download_video,
    fetch_saved_videos,
    get_next_pending_video,
    get_pipeline_status,
    run_encode_video,
    run_enhance_frames,
    run_extract_frames,
    run_upload_video,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("agent")


# ── Tool definitions (Claude sees these) ──────────────────────────────────────

TOOLS: list[Dict[str, Any]] = [
    {
        "name": "fetch_saved_videos",
        "description": (
            "Fetch the user's YouTube Saved/Watch-Later playlist and register "
            "any new videos into the pipeline state as 'pending'. "
            "Call this first to discover what needs to be processed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "playlist_id": {
                    "type": "string",
                    "description": "YouTube playlist ID. Defaults to 'WL' (Watch Later/Saved).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_pipeline_status",
        "description": (
            "Return a summary of all videos and their current pipeline stage "
            "(pending / downloading / downloaded / extracting_frames / "
            "frames_extracted / enhancing / enhanced / encoding / encoded / "
            "uploading / complete / failed). Use to decide what to do next."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "download_video",
        "description": (
            "Download a single YouTube video at the highest available resolution. "
            "Idempotent — safe to call again if interrupted. "
            "Transitions state: pending → downloading → downloaded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The YouTube video ID (11-character string).",
                }
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "run_extract_frames",
        "description": (
            "Extract all frames from a downloaded video as PNG images. "
            "Resumes from where it left off if interrupted. "
            "Transitions state: downloaded → extracting_frames → frames_extracted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"}
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "run_enhance_frames",
        "description": (
            "Upscale all extracted frames to 4K using Real-ESRGAN on the GPU. "
            "Skips already-enhanced frames for crash recovery. "
            "Uses model 'realesr-generalvideoV3' by default (best for video). "
            "Transitions state: frames_extracted → enhancing → enhanced."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "model_name": {
                    "type": "string",
                    "description": (
                        "Real-ESRGAN model. Options: "
                        "'realesr-generalvideoV3' (video, fast), "
                        "'RealESRGAN_x4plus' (general high quality), "
                        "'RealESRGAN_x4plus_anime_6B' (anime content)."
                    ),
                    "enum": [
                        "realesr-generalvideoV3",
                        "RealESRGAN_x4plus",
                        "RealESRGAN_x4plus_anime_6B",
                    ],
                },
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "run_encode_video",
        "description": (
            "Reconstruct the 4K MP4 from enhanced frames + original audio using "
            "H.265/HEVC encoding. Idempotent. "
            "Transitions state: enhanced → encoding → encoded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"}
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "run_upload_video",
        "description": (
            "Upload the encoded 4K video to the 'Down Memory Lane' YouTube channel "
            "with an AI-generated title and description. "
            "Uses resumable upload — safe to retry on network errors. "
            "Transitions state: encoded → uploading → complete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"}
            },
            "required": ["video_id"],
        },
    },
]

# ── Tool executor ──────────────────────────────────────────────────────────────

def execute_tool(name: str, tool_input: Dict[str, Any]) -> str:
    """Dispatch a tool call and return the result as a JSON string."""
    logger.info("Tool call: %s(%s)", name, json.dumps(tool_input))

    try:
        if name == "fetch_saved_videos":
            result = fetch_saved_videos(tool_input.get("playlist_id", SAVED_PLAYLIST_ID))
        elif name == "get_pipeline_status":
            result = get_pipeline_status()
        elif name == "download_video":
            result = download_video(tool_input["video_id"])
        elif name == "run_extract_frames":
            result = run_extract_frames(tool_input["video_id"])
        elif name == "run_enhance_frames":
            result = run_enhance_frames(
                tool_input["video_id"],
                model_name=tool_input.get("model_name", REALESRGAN_MODEL),
            )
        elif name == "run_encode_video":
            result = run_encode_video(tool_input["video_id"])
        elif name == "run_upload_video":
            result = run_upload_video(tool_input["video_id"])
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}
    except Exception as exc:
        logger.exception("Tool %s raised an exception", name)
        result = {"success": False, "error": str(exc)}

    return json.dumps(result, default=str)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an autonomous video restoration agent for the "Down Memory Lane" project.

Your mission: curate, denoise, upscale, and publish YouTube videos in stunning 4K.

## Pipeline stages per video:
1. fetch_saved_videos        — Discover what's in the user's YouTube Saved playlist
2. download_video            — Download at highest available resolution
3. run_extract_frames        — Extract all frames as PNGs
4. run_enhance_frames        — Upscale each frame 4× with Real-ESRGAN (GPU)
5. run_encode_video          — Reconstruct 4K H.265 MP4 with original audio
6. run_upload_video          — Upload to the "Down Memory Lane" YouTube channel

## Rules:
- ALWAYS call get_pipeline_status before starting to understand current state
- Process videos one complete pipeline at a time (all 5 stages) before moving to the next
- For failed videos: retry once, then skip with a log message
- Prefer the 'realesr-generalvideoV3' model for most videos (it's temporally stable)
- Switch to 'RealESRGAN_x4plus' for very degraded or historic footage
- Switch to 'RealESRGAN_x4plus_anime_6B' for animated/cartoon content
- Never call run_extract_frames unless download_video succeeded
- Never call run_enhance_frames unless run_extract_frames succeeded
- Report progress clearly after each tool call
- When all videos are complete, summarize the results
""".strip()


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_agent(initial_prompt: str) -> None:
    """Run the Claude agent with tool use until it finishes."""
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set in environment")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": initial_prompt}]

    logger.info("Starting agent with model: %s", CLAUDE_MODEL)

    while True:
        # Stream response for long operations
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        # Show text output
        for block in response.content:
            if hasattr(block, "text"):
                print(f"\n[Agent]: {block.text}")

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Stop conditions
        if response.stop_reason == "end_turn":
            logger.info("Agent finished.")
            break

        if response.stop_reason != "tool_use":
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_str = execute_tool(block.name, block.input)
                logger.info("Tool %s → %s", block.name, result_str[:200])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    }
                )

        messages.append({"role": "user", "content": tool_results})


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube 4K Enhancer Agent")
    parser.add_argument("--status", action="store_true", help="Show pipeline status and exit")
    parser.add_argument("--video", metavar="VIDEO_ID", help="Process one specific video ID")
    parser.add_argument(
        "--playlist",
        default=SAVED_PLAYLIST_ID,
        help="YouTube playlist ID to process (default: WL = Watch Later / Saved)",
    )
    args = parser.parse_args()

    if args.status:
        status = get_pipeline_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.video:
        prompt = (
            f"Process the single video with ID '{args.video}' through the full pipeline: "
            "download → extract frames → enhance frames → encode → upload. "
            "Check its current stage first and resume from where it left off."
        )
    else:
        prompt = (
            f"Start the full restoration pipeline for playlist '{args.playlist}'. "
            "First call fetch_saved_videos to discover all videos, "
            "then call get_pipeline_status to see what needs work. "
            "Process each pending (or failed) video through the complete pipeline: "
            "download → extract_frames → enhance_frames → encode → upload. "
            "Report progress after each video completes."
        )

    run_agent(prompt)


if __name__ == "__main__":
    main()
