import torch

from src.audio import extract_audio
from src.diarization import diarize
from src.transcript import transcribe


def main():

    # Entry point of the video anonymization pipeline
    print("VIDEO ANONYMIZATION PIPELINE")

    # Checks available hardware to determine whether GPU acceleration can be used
    if torch.cuda.is_available():
        print("GPU detected -> using CUDA")
    else:
        print("No GPU -> using CPU")

    # Step 1: Audio extraction from the input video file
    print("\n[1/3] Extract audio")
    extract_audio()

    # Step 2: Speaker diarization to segment audio by speaker identity and time
    print("\n[2/3] Speaker diarization")
    diarize()

     # Step 3: Speech transcription using Whisper-based model
    print("\n[3/3] Whisper transcription")
    transcribe()

    # Indicates that the full pipeline has completed successfully
    print("\nDONE")


if __name__ == "__main__":
    main()