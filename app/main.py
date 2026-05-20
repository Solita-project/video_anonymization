import streamlit as st
import os
import subprocess
import json
import sys

from video_anonymization.src.audio import VIDEO_PATH

# Paths
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

# Importing functions from other modules
st.page_config(page_title="Video Anonymization", layout="wide")

st.title("Video Anonymization App")
st.write("This app anonymizes videos by extracting audio, transcribing it, performing speaker diarization, and generating new audio with text-to-speech.")

# File uploader for video input
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    with open(VIDEO_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Video uploaded successfully: {uploaded_file.name}")
    st.video(VIDEO_PATH)


