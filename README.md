# Video anonymization for health care use

## Structure

- `venvs/core` → core logic
- `venvs/whisperx` → speech-to-text (WhisperX)
- `venvs/speaker` → speaker diarization (Pyannote)
- `venvs/tts` → text-to-speech (Piper + Chatterbox)
- `venvs/video` → video processing (YOLO)

# Project Setup Guide

This project uses multiple isolated Python virtual environments (`venvs`) to avoid dependency conflicts between AI libraries.

Project structure:

```text
project/
│
├── data/
├── src/
├── venvs/
├── requirements/
├── run/
├── tools/
└── README.md
```

---

# Step 1: Create venvs folder

Open terminal (Git Bash on Windows or Terminal on macOS/Linux):

```bash
mkdir venvs
```

---

# Step 2: Create virtual environments

## Windows (Git Bash)

### Core

```bash
python -m venv venvs/core
```

### WhisperX CPU

```bash
python -m venv venvs/whisperx_cpu
```

### WhisperX GPU

```bash
python -m venv venvs/whisperx_gpu
```

### Speaker CPU

```bash
python -m venv venvs/speaker_cpu
```

### Speaker GPU

```bash
python -m venv venvs/speaker_gpu
```

### TTS

```bash
python -m venv venvs/tts
```

### Video CPU

```bash
python -m venv venvs/video_cpu
```

### Video GPU

```bash
python -m venv venvs/video_gpu
```

---

## macOS / Linux

### Core

```bash
python3 -m venv venvs/core
```

### WhisperX CPU

```bash
python3 -m venv venvs/whisperx_cpu
```

### WhisperX GPU

```bash
python3 -m venv venvs/whisperx_gpu
```

### Speaker CPU

```bash
python3 -m venv venvs/speaker_cpu
```

### Speaker GPU

```bash
python3 -m venv venvs/speaker_gpu
```

### TTS

```bash
python3 -m venv venvs/tts
```

### Video CPU

```bash
python3 -m venv venvs/video_cpu
```

### Video GPU

```bash
python3 -m venv venvs/video_gpu
```

---

# Step 3: Install dependencies

## Windows (Git Bash)

Activate venv:

```bash
source venvs/<venv-name>/Scripts/activate
```

Example:

```bash
source venvs/whisperx_cpu/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements/whisperx/cpu.txt
```

Deactivate:

```bash
deactivate
```

Repeat for every venv.

Examples:

### WhisperX GPU

```bash
source venvs/whisperx_gpu/Scripts/activate
pip install -r requirements/whisperx/gpu.txt
deactivate
```

### Speaker CPU

```bash
source venvs/speaker_cpu/Scripts/activate
pip install -r requirements/speaker/cpu.txt
deactivate
```

### Speaker GPU

```bash
source venvs/speaker_gpu/Scripts/activate
pip install -r requirements/speaker/gpu.txt
deactivate
```

### TTS

```bash
source venvs/tts/Scripts/activate
pip install -r requirements/tts/cpu.txt
deactivate
```

### Video CPU

```bash
source venvs/video_cpu/Scripts/activate
pip install -r requirements/video/cpu.txt
deactivate
```

### Video GPU

```bash
source venvs/video_gpu/Scripts/activate
pip install -r requirements/video/gpu.txt
deactivate
```

---

## macOS / Linux

Activate:

```bash
source venvs/<venv-name>/bin/activate
```

Example:

```bash
source venvs/whisperx_cpu/bin/activate
```

Install:

```bash
pip install -r requirements/whisperx/cpu.txt
```

Deactivate:

```bash
deactivate
```

---

# Step 4: Create run folder

Create:

```bash
mkdir run
```

Structure:

```text
run/
│
├── run_whisperx.py
├── run_speaker.py
├── run_tts.py
├── run_video.py
└── run_pipeline.py
```

---

# Step 5: Add run files

run/run_whisperx.py

```python
import subprocess
import torch
import platform

gpu=torch.cuda.is_available()

if platform.system()=="Windows":

    python_path=(
        "venvs/whisperx_gpu/Scripts/python.exe"
        if gpu
        else "venvs/whisperx_cpu/Scripts/python.exe"
    )

else:

    python_path=(
        "venvs/whisperx_gpu/bin/python"
        if gpu
        else "venvs/whisperx_cpu/bin/python"
    )

subprocess.run([
    python_path,
    "src/speech/transcript.py"
])
```

---

run/run_speaker.py

```python
import subprocess
import torch
import platform

gpu=torch.cuda.is_available()

if platform.system()=="Windows":

    python_path=(
        "venvs/speaker_gpu/Scripts/python.exe"
        if gpu
        else "venvs/speaker_cpu/Scripts/python.exe"
    )

else:

    python_path=(
        "venvs/speaker_gpu/bin/python"
        if gpu
        else "venvs/speaker_cpu/bin/python"
    )

subprocess.run([
    python_path,
    "src/speech/diarization.py"
])
```

---

run/run_tts.py

```python
import subprocess
import platform

if platform.system()=="Windows":
    python_path="venvs/tts/Scripts/python.exe"
else:
    python_path="venvs/tts/bin/python"

subprocess.run([
    python_path,
    "src/tts/tts.py"
])
```

---

run/run_video.py

```python
import subprocess
import torch
import platform

gpu=torch.cuda.is_available()

if platform.system()=="Windows":

    python_path=(
        "venvs/video_gpu/Scripts/python.exe"
        if gpu
        else "venvs/video_cpu/Scripts/python.exe"
    )

else:

    python_path=(
        "venvs/video_gpu/bin/python"
        if gpu
        else "venvs/video_cpu/bin/python"
    )

subprocess.run([
    python_path,
    "src/video/video.py"
])
```

---

run/run_pipeline.py

```python
import subprocess

subprocess.run(["python","run/run_whisperx.py"])
subprocess.run(["python","run/run_speaker.py"])
subprocess.run(["python","run/run_tts.py"])
subprocess.run(["python","run/run_video.py"])
```

---

# Step 6: Run the project

Run full pipeline:

```bash
python run/run_pipeline.py
```

Run individual steps:

```bash
python run/run_whisperx.py
```

```bash
python run/run_video.py
```

---

# Notes

- GPU environments are automatically selected if CUDA is available.
- CPU environments are used as fallback.
- Each AI stack is isolated in its own venv.
- Avoid mixing WhisperX and Pyannote in the same venv.
- Data moves through files inside `data/`.
