# !/usr/bin/env bash
set -e

# Creates Windows CPU virtual environments and installs dependencies.
# Usage in Git Bash:
# chmod +x starter/setup_windows_cpu.sh
# ./starter/setup_windows_cpu.sh

# Move to project root
cd "$(dirname "$0")/.."

# Use Python 3.10 from Windows Python Launcher
PYTHON_PATH="$(py -3.10 -c "import sys; print(sys.executable)")"

# Create virtual environments
"$PYTHON_PATH" -m venv venvs/core
"$PYTHON_PATH" -m venv venvs/speaker
"$PYTHON_PATH" -m venv venvs/transcript
"$PYTHON_PATH" -m venv venvs/tts
"$PYTHON_PATH" -m venv venvs/video

# Upgrade pip
venvs/core/Scripts/python.exe -m pip install --upgrade pip
venvs/speaker/Scripts/python.exe -m pip install --upgrade pip
venvs/transcript/Scripts/python.exe -m pip install --upgrade pip
venvs/tts/Scripts/python.exe -m pip install --upgrade pip
venvs/video/Scripts/python.exe -m pip install --upgrade pip

# Install core dependencies
venvs/core/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venvs/core/Scripts/python.exe -m pip install -r requirements/core/base.txt

# Install speaker dependencies
venvs/speaker/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/speaker/Scripts/python.exe -m pip install -r requirements/speaker/cpu.txt
venvs/speaker/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transcript dependencies
venvs/transcript/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/transcript/Scripts/python.exe -m pip install -r requirements/transcript/cpu.txt
venvs/transcript/Scripts/python.exe -m spacy download fi_core_news_sm
venvs/transcript/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TTS dependencies
venvs/tts/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/tts/Scripts/python.exe -m pip install -r requirements/tts/cpu.txt
venvs/tts/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install video dependencies
venvs/video/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
venvs/video/Scripts/python.exe -m pip install -r requirements/video/cpu.txt
venvs/video/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu