#!/usr/bin/env bash
set -e

# Creates Linux CPU virtual environments and installs dependencies.
# Usage:
# chmod +x starter/setup_linux_cpu.sh
# ./starter/setup_linux_cpu.sh

# Move to project root
cd "$(dirname "$0")/.."

# Use Python 3.10
PYTHON=python3.10

# Create virtual environments
$PYTHON -m venv venvs/core
$PYTHON -m venv venvs/speaker
$PYTHON -m venv venvs/transcript
$PYTHON -m venv venvs/tts
$PYTHON -m venv venvs/video

# Upgrade pip
venvs/core/bin/python -m pip install --upgrade pip
venvs/speaker/bin/python -m pip install --upgrade pip
venvs/transcript/bin/python -m pip install --upgrade pip
venvs/tts/bin/python -m pip install --upgrade pip
venvs/video/bin/python -m pip install --upgrade pip

# Install core dependencies
venvs/core/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venvs/core/bin/python -m pip install -r requirements/core/base.txt

# Install speaker dependencies
venvs/speaker/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/speaker/bin/python -m pip install -r requirements/speaker/cpu.txt
venvs/speaker/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transcript dependencies
venvs/transcript/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/transcript/bin/python -m pip install -r requirements/transcript/cpu.txt
venvs/transcript/bin/python -m spacy download fi_core_news_sm
venvs/transcript/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TTS dependencies
venvs/tts/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/tts/bin/python -m pip install -r requirements/tts/cpu.txt
venvs/tts/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install video dependencies
venvs/video/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/video/bin/python -m pip install -r requirements/video/cpu.txt
venvs/video/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu