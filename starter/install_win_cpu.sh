# script for installing the required packages for CPU usage on Windows
python -m venv venvs/core
python -m venv venvs/transcript
python -m venv venvs/speaker
python -m venv venvs/tts
python -m venv venvs/video

# Activate the virtual environments
venvs\core\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venvs\transcript\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venvs\speaker\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venvs\tts\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venvs\video\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# Install the required packages cpu
venvs\core\Scripts\python.exe -m pip install -r requirements\core\base.txt
venvs\transcript\Scripts\python.exe -m pip install -r requirements\transcript\cpu.txt
venvs\speaker\Scripts\python.exe -m pip install -r requirements\speaker\cpu.txt
venvs\tts\Scripts\python.exe -m pip install -r requirements\tts\cpu.txt
venvs\video\Scripts\python.exe -m pip install -r requirements\video\cpu.txt