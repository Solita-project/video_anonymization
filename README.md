# Video Anonymization Setup

This package contains setup scripts and dependency files for the `VIDEO_ANONYMIZATION` project.

The project uses five separate virtual environments:

```text
venvs/core        -> pipeline control and lightweight file/audio steps
venvs/transcript  -> WhisperX transcription and transcript PII redaction
venvs/speaker     -> Pyannote speaker diarization
venvs/tts         -> Chatterbox TTS
venvs/video       -> YOLO/OpenCV video processing
```

## Why the setup scripts install Torch separately

Do not put `torch`, `torchvision`, or `torchaudio` in the requirements files.

The setup scripts install the Torch stack separately so that CPU/GPU builds stay correct. This is especially important for WhisperX, because installing WhisperX dependencies can overwrite CUDA Torch with CPU Torch.

The GPU scripts now intentionally repair the Torch stack after installing requirements.

## Windows Git Bash GPU setup

From the project root:

```bash
chmod +x starter/*.sh
FORCE_RECREATE=1 PYTORCH_CUDA=cu128 ./starter/setup_windows_gpu.sh
```

If `cu128` does not work, try:

```bash
FORCE_RECREATE=1 PYTORCH_CUDA=cu126 ./starter/setup_windows_gpu.sh
```

or:

```bash
FORCE_RECREATE=1 PYTORCH_CUDA=cu118 ./starter/setup_windows_gpu.sh
```

`FORCE_RECREATE=1` deletes and recreates the virtual environments. Use it when changing between CPU and GPU setups or when fixing broken dependencies.

## Windows Git Bash CPU setup

```bash
chmod +x starter/*.sh
FORCE_RECREATE=1 ./starter/setup_windows_cpu.sh
```

## Linux GPU setup

```bash
chmod +x starter/*.sh
FORCE_RECREATE=1 PYTORCH_CUDA=cu128 ./starter/setup_linux_gpu.sh
```

## Linux CPU setup

```bash
chmod +x starter/*.sh
FORCE_RECREATE=1 ./starter/setup_linux_cpu.sh
```

## Run the full pipeline

Windows Git Bash:

```bash
source venvs/core/Scripts/activate
python run/run_pipeline.py
```

Linux/macOS:

```bash
source venvs/core/bin/activate
python run/run_pipeline.py
```

The pipeline starts from the core environment. The wrapper scripts inside `run/` call the correct Python executable for each separate virtual environment.

## Hugging Face token

Speaker diarization requires a Hugging Face token in `.env`:

```text
HF_TOKEN=your_huggingface_token_here
```
