# Upload page UI for the Streamlit app

import streamlit as st


def show_upload_section(
    reset_app_state,
    clean_old_outputs,
    save_uploaded_video,
    run_video_anonymization,
    start_transcript_process,
):
    # Show video upload section
    st.write("## Upload video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4"],
    )

    profile_name = st.selectbox(
        "Video anonymization profile",
        options=["cpr", "intubation"],
        format_func=lambda value: "CPR" if value == "cpr" else "Intubation",
        key="video_profile",
    )

    if not uploaded_file:
        return

    st.write(f"Selected file: `{uploaded_file.name}`")

    if st.button("Anonymize video"):
        reset_app_state()
        clean_old_outputs()
        save_uploaded_video(uploaded_file)

        with st.spinner("Running video anonymization..."):
            run_video_anonymization(profile_name)

        start_transcript_process()

        st.session_state.page = "processing"
        st.rerun()
