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