!/bin/bash
set -e

python -m venv venvs/app_env
python -m venv venvs/audio_env
python -m venv venvs/diarization_env
python -m venv venvs/transcript_env
python -m venv venvs/video_env

# Activate the virtual environments and install the required packages cpu
venvs/app_env/bin/python -m pip install --upgrade pip
venvs/audio_env/bin/python -m pip install --upgrade pip
venvs/diarization_env/bin/python -m pip install --upgrade pip
venvs/transcript_env/bin/python -m pip install --upgrade pip
venvs/video_env/bin/python -m pip install --upgrade pip

# Activate the virtual environments and install the required packages gpu
venvs/app_env/bin/python -m pip install -r requirements/app.txt
venvs/audio_env/bin/python -m pip install -r requirements/audio_gpu.txt
venvs/diarization_env/bin/python -m pip install -r requirements/diarization_gpu.txt
venvs/transcript_env/bin/python -m pip install -r requirements/transcript_gpu.txt
venvs/video_env/bin/python -m pip install -r requirements/video_gpu.txt