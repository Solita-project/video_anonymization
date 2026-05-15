FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 \
    torchaudio==2.11.0 \
    torchvision==0.26.0 \
    torchcodec==0.11.1

RUN pip install --no-cache-dir -r requirements.txt


COPY . .
# Temporary entrypoint for diarization.
# Later change to CMD ["python", "main.py"]
CMD ["python", "diarization.py"]
