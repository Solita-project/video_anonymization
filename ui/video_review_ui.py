# Video review UI for the Streamlit app

import streamlit as st

def show_transcript_status(status, read_transcript_log):
    # Show short transcript preparation status below the video
    if status == "running":
        st.info("Preparing transcript...")

    elif status == "ready":
        st.success("Transcript is ready.")

    elif status == "failed":
        st.error("Transcript preparation failed.")

        log_text = read_transcript_log()

        if log_text:
            with st.expander("Show technical details"):
                st.code(log_text)


def show_video_section(status, blurred_preview_file, read_transcript_log):
    # Show anonymized video and video review buttons
    st.write("## Review video anonymization")

    if blurred_preview_file.exists():
        with open(blurred_preview_file, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes, format="video/mp4")
    else:
        st.warning("Anonymized video was not found.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Video anonymization looks good"):
            st.session_state.video_approved = True
            st.session_state.show_manual_blur = False
            st.rerun()

    with col2:
        if st.button("I want to improve video anonymization"):
            st.session_state.video_approved = False
            st.session_state.show_manual_blur = True
            st.rerun()

    show_transcript_status(status, read_transcript_log)
