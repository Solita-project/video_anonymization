# Final video UI for the Streamlit app

import streamlit as st


def show_final_video(final_video_file):
    # Show final video and download button
    if not final_video_file.exists():
        return

    st.write("---")
    st.write("## Final anonymized video")

    with open(final_video_file, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes, format="video/mp4")

    st.download_button(
        label="Download video",
        data=video_bytes,
        file_name=final_video_file.name,
        mime="video/mp4",
    )
