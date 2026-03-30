#!/usr/bin/env bash
# YouTube 4K Enhancer — Unix/WSL setup script

set -euo pipefail

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing PyTorch with CUDA 12.1 (RTX 4070) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing project dependencies ==="
pip install -r requirements.txt

echo "=== Checking ffmpeg ==="
if ! command -v ffmpeg &>/dev/null; then
    echo "WARNING: ffmpeg not found. Install it:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  Windows: winget install Gyan.FFmpeg"
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. cp .env.example .env  # then fill in ANTHROPIC_API_KEY"
echo "  2. Place client_secrets.json from Google Cloud Console in project root"
echo "  3. source venv/bin/activate && python agent.py"
