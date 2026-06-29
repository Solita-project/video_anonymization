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


def show_video_report(report):
    # Show structured video review report below the anonymized video
    if report is None:
        st.info("Video review report was not found.")
        return

    review = report.get("review", {})
    summary = review.get("summary", {})
    warnings = review.get("warnings", [])
    suggested_ranges = review.get("suggested_ranges", [])

    st.write("### Video review report")

    st.write(
        f"Profile: `{report.get('profile', 'unknown')}`  "
        f"| Processed frames: `{report.get('processed_frames', 0)}` / `{report.get('total_frames', 0)}`"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Suggested ranges",
        summary.get("suggested_review_range_count", 0),
    )
    col2.metric(
        "Review seconds",
        summary.get("suggested_review_duration_seconds", 0),
    )
    col3.metric(
        "Warnings",
        summary.get("warning_count", 0),
    )

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No priority review warnings in this report.")

    if suggested_ranges:
        st.write("#### Suggested review ranges")

        for index, item in enumerate(suggested_ranges, start=1):
            st.write(
                f"{index}. `{item.get('priority', 'unknown')}` "
                f"{item.get('start_time_seconds')}s–{item.get('end_time_seconds')}s: "
                f"{item.get('reason', '')}"
            )
            st.caption(item.get("suggested_action", ""))
    else:
        st.info("No suggested review ranges.")


def show_video_section(status, blurred_preview_file, load_video_report, read_transcript_log):
    # Show anonymized video and video review buttons
    st.write("## Review video anonymization")

    if blurred_preview_file.exists():
        with open(blurred_preview_file, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes, format="video/mp4")
    else:
        st.warning("Anonymized video was not found.")
        return

    show_video_report(load_video_report())

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
