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
    "src/video.py"
])