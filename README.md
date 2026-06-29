# Video Anonymization

This project is a local video anonymization pipeline with a Streamlit user
interface. It anonymizes both the video image and the spoken audio/transcript.

The current implementation is built around Finnish speech and Finnish personal
data patterns. It combines automatic processing with human review before the
final video is produced.

## What the pipeline does

1. Upload or place an input MP4 video at `data/input/video.mp4`.
2. Extract WAV audio to `data/input/audio.wav`.
3. Transcribe Finnish speech with WhisperX into word-level timestamps.
4. Anonymize sensitive transcript text with regex rules and spaCy.
5. Blur video regions with YOLO/OpenCV face, head and object detection.
6. Let a human reviewer check the video anonymization.
7. Optionally add manual blur rectangles for selected time ranges.
8. Let a human reviewer check automatically anonymized transcript words.
9. Optionally edit transcript segments word by word.
10. Run speaker diarization with Pyannote.
11. Merge transcript words with speaker IDs.
12. Generate anonymized replacement speech with Chatterbox TTS.
13. Merge anonymized video and anonymized audio into `final_video.mp4`.

The recommended workflow is the Streamlit app in `app.py`. A command-line
pipeline is also available, but it does not include the manual review steps.

## Project structure

```text
app.py                         Streamlit app entrypoint
ui/                            Streamlit UI sections
run/                           Wrapper scripts for separate virtual enviroments
src/                           Pipeline implementation modules
starter/                       Setup scripts for Windows/Linux and CPU/GPU
requirements/                  Dependency files per virtual environment
models/                        Local YOLO model weights, ignored by git
tools/                         Local ffmpeg binaries, ignored by git
voices/                        TTS voice prompt WAV files
data/input/                    Runtime input files, ignored by git
data/output/                   Runtime output files, ignored by git
```

## Runtime environments

The project uses separate virtual environments because the ML dependencies are
heavy and sometimes conflict with each other.

```text
venvs/core        Streamlit UI, orchestration and lightweight audio/video steps
venvs/transcript  WhisperX transcription and transcript anonymization
venvs/speaker     Pyannote speaker diarization
venvs/tts         Chatterbox TTS
venvs/video       YOLO/OpenCV video processing and manual blur processing
```

The wrapper scripts in `run/` are called from the core environment and then run
the actual workload in the correct environment.

## Required local assets

These files are expected by the current code:

```text
tools/ffmpeg.exe                         Optional on Windows if ffmpeg is not on PATH
models/yolov8s-face-lindevs.pt           Face detection model
models/yolov8s.pt                        YOLO object detection model
models/yolov8_head_medium.pt             Head detection model
voices/*.wav                             Voice prompt audio files for TTS
```

The current `voices/` folder contains several example voice prompt files. TTS
assigns speakers to voice files in sorted filename order and wraps around if
there are more speakers than voice files.

## Environment variables

Create a local `.env` file in the project root.

```text
HF_TOKEN=your_huggingface_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

`HF_TOKEN` is required for Pyannote diarization.

`GEMINI_API_KEY` required for Gemini useS.

`.env` is ignored by git.

## Setup

Use Python 3.10. The setup scripts create all five virtual environments and
install the matching dependency sets.

### Automatic setup

The automatic setup script detects Windows vs Linux and whether `nvidia-smi`
reports an NVIDIA GPU.

```bash
chmod +x starter/*.sh
./starter/setup_automatic.sh
```

### Windows Git Bash, GPU

```bash
chmod +x starter/*.sh
PYTORCH_CUDA=cu128 ./starter/setup_windows_gpu.sh
```

If that CUDA wheel set does not work for the machine, try another PyTorch CUDA
index, for example:

```bash
PYTORCH_CUDA=cu126 ./starter/setup_windows_gpu.sh
PYTORCH_CUDA=cu118 ./starter/setup_windows_gpu.sh
```

### Windows Git Bash, CPU

```bash
chmod +x starter/*.sh
./starter/setup_windows_cpu.sh
```

### Linux, GPU

```bash
chmod +x starter/*.sh
PYTORCH_CUDA=cu128 ./starter/setup_linux_gpu.sh
```

### Linux, CPU

```bash
chmod +x starter/*.sh
./starter/setup_linux_cpu.sh
```

The setup scripts assume Python 3.10 is available. On Windows they use the
Python Launcher command `py -3.10`; on Linux they use `python3.10`.

## Run the Streamlit app

Windows Git Bash:

```bash
source venvs/core/Scripts/activate
streamlit run app.py
```

Windows PowerShell:

```powershell
.\venvs\core\Scripts\Activate.ps1
streamlit run app.py
```

Linux:

```bash
source venvs/core/bin/activate
streamlit run app.py
```

The Streamlit upload limit is configured in `.streamlit/config.toml`:

```text
maxUploadSize = 5000
```

## Streamlit workflow

1. Open the app.
2. Upload an `.mp4` video.
3. Select the video anonymization profile:
   - `CPR`
   - `Intubation`
4. Click `Anonymize video`.
5. The app saves the upload as `data/input/video.mp4`.
6. Automatic video anonymization runs first.
7. Transcript preparation starts in the background:
   - audio extraction
   - WhisperX transcription
   - automatic transcript anonymization
8. Review the anonymized video preview.
9. If needed, choose `I want to improve video anonymization` and add manual
   blur rectangles.
10. Approve the video anonymization.
11. Review automatically anonymized transcript words.
12. Optionally review and edit full transcript segments.
13. Click `Continue anonymization`.
14. The app runs diarization, transcript/speaker merging, TTS and final merge.
15. The final anonymized video is shown and can be downloaded.

## Run the full command-line pipeline

The command-line pipeline runs without Streamlit review.

Windows Git Bash:

```bash
source venvs/core/Scripts/activate
python run/run_pipeline.py
```

Linux:

```bash
source venvs/core/bin/activate
python run/run_pipeline.py
```

Steps executed by `run/run_pipeline.py`:

```text
1. src/audio.py
2. run/run_transcript.py
3. run/run_anonymize.py
4. run/run_speaker.py
5. src/merged.py
6. run/run_tts.py
7. run/run_video.py
8. src/final_merge.py
```

Note that this path skips manual video review, manual blur and transcript review.
It also uses the default video profile and current default frame limit.

## Run individual steps

From the project root with the core environment active:

```bash
python src/audio.py
python run/run_transcript.py
python run/run_anonymize.py
python run/run_speaker.py
python src/merged.py
python run/run_tts.py
python run/run_video.py --profile cpr
python run/run_video.py --profile intubation
python src/final_merge.py
```

The wrappers choose the correct environment for transcript, speaker, TTS and
video processing.

## Main input and output files

```text
data/input/video.mp4                    Uploaded or manually added input video
data/input/audio.wav                    Extracted audio

data/output/transcription.json          Original WhisperX word-level transcript
data/output/cleaned_transcription.json  Automatically and manually anonymized transcript
data/output/transcript_background.log   Streamlit background transcript log

data/output/video_blurred.mp4           Blurred video without final audio
data/output/video_blurred_preview.mp4   Browser-friendly preview video
data/output/video_debug_boxes.mp4       Debug video with detection boxes
data/output/video_report.json           Video review report

data/output/manual_blur_frame.jpg       Frame extracted for manual blur drawing
data/output/manual_video_blurs.json     Saved manual blur annotations

data/output/diarization.json            Speaker diarization output
data/output/final_transcript.json       Transcript merged with speaker labels
data/output/clean_audio.wav             Generated anonymized speech
data/output/final_video.mp4             Final anonymized video
```

## Streamlit cleanup behavior

When a new Streamlit session starts, the app clears previous runtime files from:

```text
data/input/
data/output/
```

It also clears those folders when a new video is uploaded and anonymization is
started.

Download or copy any result you need before starting a new session or processing
a new video.

## Git-ignored runtime files

The following are intentionally ignored by git:

```text
tools/
venvs/
data/
.cache/
.env
models/*.pt
__pycache__/
*.pyc
```
