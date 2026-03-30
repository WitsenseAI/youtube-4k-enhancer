@echo off
REM YouTube 4K Enhancer — Windows setup script
REM Run once to install all dependencies

echo === Creating virtual environment ===
python -m venv venv
call venv\Scripts\activate.bat

echo === Installing PyTorch with CUDA 12.1 (RTX 4070) ===
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo === Installing project dependencies ===
pip install -r requirements.txt

echo === Checking ffmpeg ===
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg not found in PATH.
    echo Install via: winget install Gyan.FFmpeg
    echo Or download from: https://www.gyan.dev/ffmpeg/builds/
)

echo === Setup complete ===
echo.
echo Next steps:
echo   1. Copy .env.example to .env and fill in your ANTHROPIC_API_KEY
echo   2. Download OAuth2 client_secrets.json from Google Cloud Console
echo   3. Run: python agent.py
echo.
pause
