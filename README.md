# Video Anonymization Setup

This package contains setup scripts, dependency files and a Streamlit user interface for the `VIDEO_ANONYMIZATION` project.

The project anonymizes videos by:

1. extracting audio from the input video
2. transcribing speech with WhisperX
3. automatically anonymizing sensitive text
4. allowing a human reviewer to check and edit the anonymized transcript
5. detecting speakers
6. generating new anonymized speech with TTS
7. blurring/anonymizing the video
8. merging the anonymized audio and video into `final_video.mp4`

## Virtual environments

The project uses five separate virtual environments:

```text
venvs/core        -> Streamlit UI, pipeline control and lightweight file/audio steps
venvs/transcript  -> WhisperX transcription and transcript anonymization
venvs/speaker     -> Pyannote speaker diarization
venvs/tts         -> Chatterbox TTS
venvs/video       -> YOLO/OpenCV video processing
```

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

## Run the Streamlit app

Windows Git Bash:

```bash
source venvs/core/Scripts/activate
streamlit run app.py
```

Linux/macOS:

```bash
source venvs/core/bin/activate
streamlit run app.py
```

The Streamlit app lets the user:

1. upload an `.mp4` video
2. run automatic transcription and anonymization
3. manually review `cleaned_transcription.json`
4. edit words or replace them with anonymization labels
5. continue the anonymization pipeline
6. watch the final anonymized video in the browser
7. download the final anonymized video

The final video is saved to:

```text
data/output/final_video.mp4
```

## Streamlit cleanup behavior

The Streamlit app removes old input and output files when a new Streamlit session starts. This prevents old videos, transcripts or output files from appearing when the app is opened again later.

The app also removes old files when a new video is uploaded and anonymization is started.

This affects files inside:

```text
data/input/
data/output/
```

If you need to keep a generated video or transcript, download or copy it before starting a new session or processing a new video.

## Run the full pipeline without Streamlit

The full pipeline can also be run from the command line.

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

This runs the pipeline without the Streamlit manual review interface.

## Hugging Face token

Speaker diarization requires a Hugging Face token in `.env`:

```text
HF_TOKEN=your_huggingface_token_here
```

## Output files

```text
data/input/video.mp4                   -> uploaded or manually added input video
data/input/audio.wav                   -> extracted audio
data/output/transcription.json         -> original WhisperX transcription
data/output/cleaned_transcription.json -> automatically and manually anonymized transcript
data/output/diarization.json           -> speaker diarization result
data/output/final_transcript.json      -> merged transcript and speaker data
data/output/clean_audio.wav            -> generated anonymized speech
data/output/video_blurred.mp4          -> anonymized video without final audio
data/output/final_video.mp4            -> final anonymized video
```
