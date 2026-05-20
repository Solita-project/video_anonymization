#!/bin/bash
python -m venv venvs/app_env
python -m venv venvs/audio_env
python -m venv venvs/diarization_env
python -m venv venvs/transcript_env
python -m venv venvs/video_env

# Activate the virtual environments and install the required packages cpu
venvs\app_env\Scripts\python.exe -m pip install requirements\app.txt
venvs\audio_env\Scripts\python.exe -m pip install requirements\audio_cpu.txt
venvs\diarization_env\Scripts\python.exe -m pip install requirements\diarization_cpu.txt
venvs\transcript_env\Scripts\python.exe -m pip install requirements\transcript_cpu.txt
venvs\video_env\Scripts\python.exe -m pip install requirements\video_cpu.txt

# Activate the virtual environments and install the required packages gpu
venvs\app_env\Scripts\python.exe -m pip install -r requirements\app.txt
venvs\audio_env\Scripts\python.exe -m pip install -r requirements\audio_gpu.txt
venvs\diarization_env\Scripts\python.exe -m pip install -r requirements\diarization_gpu.txt
venvs\transcript_env\Scripts\python.exe -m pip install -r requirements\transcript_gpu.txt
venvs\video_env\Scripts\python.exe -m pip install -r requirements\video_gpu.txt