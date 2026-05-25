#!/usr/bin/env bash
set -e

# Creates Windows GPU virtual environments and installs dependencies.
# Usage in Git Bash:
# chmod +x starter/setup_windows_gpu.sh
# PYTORCH_CUDA=cu128 ./starter/setup_windows_gpu.sh

# Move to project root
cd "$(dirname "$0")/.."

# Use Python 3.10 from Windows Python Launcher
PYTHON_PATH="$(py -3.10 -c "import sys; print(sys.executable)")"

# Select CUDA version
CUDA=${PYTORCH_CUDA:-cu128}
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"

# Create virtual environments
"$PYTHON_PATH" -m venv venvs/core
"$PYTHON_PATH" -m venv venvs/speaker
"$PYTHON_PATH" -m venv venvs/transcript
"$PYTHON_PATH" -m venv venvs/tts
"$PYTHON_PATH" -m venv venvs/video

# Upgrade pip tools
venvs/core/Scripts/python.exe -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/speaker/Scripts/python.exe -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/transcript/Scripts/python.exe -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/tts/Scripts/python.exe -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/video/Scripts/python.exe -m pip install --upgrade "pip<27" "setuptools<82" wheel

# Install core dependencies
venvs/core/Scripts/python.exe -m pip install torch --index-url "$TORCH_INDEX"
venvs/core/Scripts/python.exe -m pip install -r requirements/core/base.txt
venvs/core/Scripts/python.exe -m pip install --force-reinstall torch --index-url "$TORCH_INDEX"

# Install speaker dependencies
venvs/speaker/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/speaker/Scripts/python.exe -m pip install -r requirements/speaker/gpu.txt
venvs/speaker/Scripts/python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Install transcript dependencies
venvs/transcript/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/transcript/Scripts/python.exe -m pip install -r requirements/transcript/gpu.txt
venvs/transcript/Scripts/python.exe -m spacy download fi_core_news_sm
venvs/transcript/Scripts/python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Install TTS dependencies
venvs/tts/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/tts/Scripts/python.exe -m pip install -r requirements/tts/gpu.txt
venvs/tts/Scripts/python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Install video dependencies
venvs/video/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/video/Scripts/python.exe -m pip install -r requirements/video/gpu.txt
venvs/video/Scripts/python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"