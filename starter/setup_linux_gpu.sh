#!/usr/bin/env bash
set -e

# Creates Linux GPU virtual environments and installs dependencies.
# Usage:
# chmod +x starter/setup_linux_gpu.sh
# PYTORCH_CUDA=cu128 ./starter/setup_linux_gpu.sh

# Move to project root
cd "$(dirname "$0")/.."

# Use Python 3.10
PYTHON=python3.10

# Select CUDA version
CUDA=${PYTORCH_CUDA:-cu128}
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"

# Create virtual environments
$PYTHON -m venv venvs/core
$PYTHON -m venv venvs/speaker
$PYTHON -m venv venvs/transcript
$PYTHON -m venv venvs/tts
$PYTHON -m venv venvs/video

# Upgrade pip tools
venvs/core/bin/python -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/speaker/bin/python -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/transcript/bin/python -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/tts/bin/python -m pip install --upgrade "pip<27" "setuptools<82" wheel
venvs/video/bin/python -m pip install --upgrade "pip<27" "setuptools<82" wheel

# Install core dependencies
venvs/core/bin/python -m pip install torch --index-url "$TORCH_INDEX"
venvs/core/bin/python -m pip install -r requirements/core/base.txt
venvs/core/bin/python -m pip install --force-reinstall torch --index-url "$TORCH_INDEX"

# Install speaker dependencies
venvs/speaker/bin/python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/speaker/bin/python -m pip install -r requirements/speaker/gpu.txt
venvs/speaker/bin/python -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Install transcript dependencies
venvs/transcript/bin/python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/transcript/bin/python -m pip install -r requirements/transcript/gpu.txt
venvs/transcript/bin/python -m spacy download fi_core_news_sm
venvs/transcript/bin/python -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Install TTS dependencies
venvs/tts/bin/python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/tts/bin/python -m pip install -r requirements/tts/gpu.txt
venvs/tts/bin/python -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Install video dependencies
venvs/video/bin/python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
venvs/video/bin/python -m pip install -r requirements/video/gpu.txt
venvs/video/bin/python -m pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"