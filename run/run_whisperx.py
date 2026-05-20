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