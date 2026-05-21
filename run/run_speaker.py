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
    "src/diarization.py"
])