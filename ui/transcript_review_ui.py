# Streamlit transcript review UI
# This module keeps transcript review logic out of app.py.
# The first step reviews only automatically anonymized words.
# The second step allows optional full transcript review by segment.

import html
import json
import pandas as pd
import streamlit as st


def load_transcript(transcript_file):
    # Load transcript JSON file
    with open(transcript_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_transcript(transcript_file, data):
    # Save transcript JSON file
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def initialize_review_state():
    # Initialize transcript review state
    if "review_data" not in st.session_state:
        st.session_state.review_data = None

    if "original_review_data" not in st.session_state:
        st.session_state.original_review_data = None

    if "automatic_review_done" not in st.session_state:
        st.session_state.automatic_review_done = False

    if "editing_segment" not in st.session_state:
        st.session_state.editing_segment = None

    if "review_editor_version" not in st.session_state:
        st.session_state.review_editor_version = 0


def clean_text(words):
    # Build readable text from word list
    text_words = []

    for word in words:
        word_text = word.get("word", "").strip()

        if word_text:
            text_words.append(word_text)

    text = " ".join(text_words)

    # Fix spaces before punctuation
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")

    return text.strip()


def rebuild_segment_text(segment):
    # Rebuild segment text from word list
    segment["text"] = clean_text(segment.get("words", []))


def get_original_segment(original_data, index):
    # Get matching original segment if it exists
    if not original_data:
        return None

    if index >= len(original_data):
        return None

    return original_data[index]


def get_original_word(original_segment, word_index, fallback_word):
    # Get original word from the original transcript
    # If original transcript is missing or shorter, use fallbackS
    if not original_segment:
        return fallback_word

    original_words = original_segment.get("words", [])

    if word_index >= len(original_words):
        return fallback_word

    return original_words[word_index].get("word", fallback_word)


def make_segment_text(segment):
    # Return readable segment text
    return segment.get("text", "").strip()


def make_segment_title(segment):
    # Use only the segment text as the expander title
    text = make_segment_text(segment)

    if not text:
        return "(empty segment)"

    return text[:160] + "..." if len(text) > 160 else text


def format_action_label(value):
    # Make RESTORE_ORIGINAL easier to read in the UI
    if value == "RESTORE_ORIGINAL":
        return "RESTORE ORIGINAL"

    return value


def apply_word_action(original_word, anonymized_word, action):
    # Apply the selected action to one word
    if action == "KEEP":
        return anonymized_word

    if action == "RESTORE_ORIGINAL":
        return original_word

    if action == "REMOVE":
        return ""

    # If action is a label, use that label as the replacement word
    return action


def show_automatic_review_instructions():
    # Show automatic anonymization review instructions in a visually separate help box
    with st.container(border=True):
        st.markdown("#### ℹ️ Automatic anonymization review instructions")

        st.markdown(
            """
            This section shows only the words that were changed automatically by the transcript anonymization step.

            Use this table to quickly check whether the automatic anonymization made the right decision.

            **Original word** shows what the word was before automatic anonymization.

            **Automatic suggestion** shows what the anonymization step changed the word into.

            **Action** lets you decide what should happen to the word:

            - **KEEP** keeps the automatic anonymization suggestion.
            - **RESTORE_ORIGINAL** changes the word back to the original word.
            - **REMOVE** removes the word from the final transcript.
            - An anonymization label, such as **NIMI**, **SIJAINTI** or **ORGANISAATIO**, replaces the word with that label.

            **Final word** shows what will actually be saved after your selected action.

            When all automatically anonymized words look correct, click **Apply automatic anonymization review**.
            """
        )


def show_transcript_review_instructions():
    # Show transcript review instructions in a visually separate help box
    with st.container(border=True):
        st.markdown("#### ℹ️ Transcript review instructions")

        st.markdown(
            """
            Open any transcript segment below to read it.
            You do not need to review every segment before continuing.

            **Current anonymized text** is shown first.

            **Original text** is shown below it in light grey so you can compare the anonymized version to the original transcript.

            Click **Edit this segment** if you want to edit that segment word by word.

            In edit mode:
            - **Original word** shows the original transcription.
            - **Anonymized word** shows what will be used in the final transcript.
            - **Action** controls whether the word is kept, restored, removed or replaced with a label.

            Click **Save changes** to save the segment and close the editor.
            """
        )


def find_automatic_anonymization_rows(review_data, original_data, sensitive_labels):
    # Find words that were automatically changed into anonymization labels.
    rows = []

    for segment_index, reviewed_segment in enumerate(review_data):
        original_segment = get_original_segment(original_data, segment_index)
        reviewed_words = reviewed_segment.get("words", [])

        for word_index, word in enumerate(reviewed_words):
            anonymized_word = word.get("word", "")

            if anonymized_word not in sensitive_labels:
                continue

            original_word = get_original_word(
                original_segment=original_segment,
                word_index=word_index,
                fallback_word=anonymized_word,
            )

            rows.append({
                "segment_index": segment_index,
                "word_index": word_index,
                "original_word": original_word,
                "anonymized_word": anonymized_word,
            })

    return rows


def get_auto_action_key(row):
    # Build stable widget key for automatic review action
    version = st.session_state.review_editor_version
    segment_index = row["segment_index"]
    word_index = row["word_index"]

    return f"auto_action_{version}_{segment_index}_{word_index}"


def show_automatic_review_row(row, replacements):
    # Show one automatically anonymized word review row
    action_key = get_auto_action_key(row)

    original_word = row["original_word"]
    anonymized_word = row["anonymized_word"]

    col1, col2, col3, col4 = st.columns([1.5, 1.5, 2, 1.5])

    with col1:
        st.write(original_word)

    with col2:
        st.write(anonymized_word)

    with col3:
        action = st.selectbox(
            "Action",
            options=replacements,
            index=replacements.index("KEEP") if "KEEP" in replacements else 0,
            key=action_key,
            label_visibility="collapsed",
            format_func=format_action_label,
        )

    final_word = apply_word_action(
        original_word=original_word,
        anonymized_word=anonymized_word,
        action=action,
    )

    with col4:
        if final_word:
            st.write(final_word)
        else:
            st.write("REMOVED")

    return action


def apply_automatic_review_rows(rows, transcript_file):
    # Apply selected actions from automatic anonymization review
    for row in rows:
        segment_index = row["segment_index"]
        word_index = row["word_index"]
        action_key = get_auto_action_key(row)

        action = st.session_state.get(action_key, "KEEP")

        original_word = row["original_word"]
        anonymized_word = row["anonymized_word"]

        new_word = apply_word_action(
            original_word=original_word,
            anonymized_word=anonymized_word,
            action=action,
        )

        words = st.session_state.review_data[segment_index].get("words", [])

        if word_index >= len(words):
            continue

        words[word_index]["word"] = new_word
        rebuild_segment_text(st.session_state.review_data[segment_index])

    save_transcript(transcript_file, st.session_state.review_data)

    st.session_state.automatic_review_done = True
    st.session_state.review_editor_version += 1


def show_automatic_anonymization_review(transcript_file, replacements, sensitive_labels):
    # Show the first review step: automatically anonymized words
    st.write("## Review automatically anonymized words")

    show_automatic_review_instructions()

    rows = find_automatic_anonymization_rows(
        review_data=st.session_state.review_data,
        original_data=st.session_state.original_review_data,
        sensitive_labels=sensitive_labels,
    )

    if not rows:
        st.success("No automatically anonymized words were found.")

        if st.button("Continue to full transcript review"):
            st.session_state.automatic_review_done = True
            st.rerun()

        return

    st.write(
        "Check the automatically anonymized words below before reviewing the full transcript."
    )

    header_col1, header_col2, header_col3, header_col4 = st.columns([1.5, 1.5, 2, 1.5])

    with header_col1:
        st.write("**Original word**")

    with header_col2:
        st.write("**Automatic suggestion**")

    with header_col3:
        st.write("**Action**")

    with header_col4:
        st.write("**Final word**")

    for row in rows:
        show_automatic_review_row(
            row=row,
            replacements=replacements,
        )

    if st.button("Apply automatic anonymization review"):
        apply_automatic_review_rows(
            rows=rows,
            transcript_file=transcript_file,
        )
        st.rerun()


def make_segment_editor_rows(anonymized_segment, original_segment, sensitive_labels):
    # Convert one segment into editable word rows
    rows = []

    anonymized_words = anonymized_segment.get("words", [])

    for index, word in enumerate(anonymized_words):
        anonymized_word = word.get("word", "")

        original_word = get_original_word(
            original_segment=original_segment,
            word_index=index,
            fallback_word=anonymized_word,
        )

        if anonymized_word in sensitive_labels:
            action = anonymized_word
        elif anonymized_word == "":
            action = "REMOVE"
        else:
            action = "KEEP"

        rows.append({
            "original_word": original_word,
            "anonymized_word": anonymized_word,
            "action": action,
        })

    return rows


def apply_segment_editor_rows(segment, edited_rows, sensitive_labels):
    # Apply manual segment edits back to one segment
    words = segment.get("words", [])
    last_sensitive_label = None

    for index, row in enumerate(edited_rows):
        if index >= len(words):
            continue

        original_word = str(row.get("original_word", "")).strip()
        anonymized_word = str(row.get("anonymized_word", "")).strip()
        action = row["action"]

        new_word = apply_word_action(
            original_word=original_word,
            anonymized_word=anonymized_word,
            action=action,
        )

        # Collapse repeated labels, for example NIMI NIMI -> NIMI
        if new_word in sensitive_labels:
            if new_word == last_sensitive_label:
                new_word = ""
            else:
                last_sensitive_label = new_word

        elif new_word:
            last_sensitive_label = None

        words[index]["word"] = new_word

    segment["words"] = words
    rebuild_segment_text(segment)


def show_original_segment_text(original_segment):
    # Show original segment text in light grey
    if not original_segment:
        return

    original_text = make_segment_text(original_segment)

    if not original_text:
        return

    safe_original_text = html.escape(original_text)

    st.markdown(
        f"""
        <div style="color: #8a8a8a; font-size: 0.95rem; margin-top: 0.5rem;">
            <strong>Original:</strong> {safe_original_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_segment_editor(index, anonymized_segment, original_segment, transcript_file, replacements, sensitive_labels):
    # Show editable table for one segment
    rows = make_segment_editor_rows(
        anonymized_segment=anonymized_segment,
        original_segment=original_segment,
        sensitive_labels=sensitive_labels,
    )

    df = pd.DataFrame(rows)

    edited_df = st.data_editor(
        df,
        hide_index=True,
        num_rows="fixed",
        disabled=["original_word"],
        column_config={
            "original_word": st.column_config.TextColumn(
                "Original word",
            ),
            "anonymized_word": st.column_config.TextColumn(
                "Anonymized word",
            ),
            "action": st.column_config.SelectboxColumn(
                "Action",
                options=replacements,
            ),
        },
        key=f"segment_editor_{st.session_state.review_editor_version}_{index}",
    )

    if st.button("Save changes", key=f"save_segment_{index}"):
        edited_rows = edited_df.to_dict("records")

        apply_segment_editor_rows(
            segment=anonymized_segment,
            edited_rows=edited_rows,
            sensitive_labels=sensitive_labels,
        )

        st.session_state.review_data[index] = anonymized_segment
        save_transcript(transcript_file, st.session_state.review_data)

        st.session_state.editing_segment = None
        st.session_state.review_editor_version += 1

        st.success("Changes saved.")
        st.rerun()


def show_segment_expander(index, anonymized_segment, original_segment, transcript_file, replacements, sensitive_labels):
    # Show one transcript segment as an expander
    title = make_segment_title(anonymized_segment)
    is_editing = st.session_state.editing_segment == index

    with st.expander(title, expanded=is_editing):
        st.write(make_segment_text(anonymized_segment))
        show_original_segment_text(original_segment)

        if is_editing:
            show_segment_editor(
                index=index,
                anonymized_segment=anonymized_segment,
                original_segment=original_segment,
                transcript_file=transcript_file,
                replacements=replacements,
                sensitive_labels=sensitive_labels,
            )
        else:
            if st.button("Edit this segment", key=f"edit_segment_{index}"):
                st.session_state.editing_segment = index
                st.session_state.review_editor_version += 1
                st.rerun()


def show_full_transcript_review(transcript_file, replacements, sensitive_labels):
    # Show the second review step: full transcript review
    st.write("---")
    st.write("## Review anonymized transcript")

    show_transcript_review_instructions()

    st.write("### Transcript segments")
    st.caption("Open a segment below to read it or edit it.")

    for index, anonymized_segment in enumerate(st.session_state.review_data):
        original_segment = get_original_segment(
            original_data=st.session_state.original_review_data,
            index=index,
        )

        show_segment_expander(
            index=index,
            anonymized_segment=anonymized_segment,
            original_segment=original_segment,
            transcript_file=transcript_file,
            replacements=replacements,
            sensitive_labels=sensitive_labels,
        )

    st.write("---")
    st.write("## Continue anonymization")

    if st.session_state.editing_segment is not None:
        st.warning("Save the currently open segment before continuing.")
        return False

    return st.button("Continue anonymization")


def show_transcript_review_section(transcript_file, original_transcript_file, replacements):
    # Show transcript review section
    # Returns True when user clicks Continue anonymization
    if not st.session_state.get("video_approved"):
        return False

    if not transcript_file.exists():
        return False

    initialize_review_state()

    if st.session_state.review_data is None:
        st.session_state.review_data = load_transcript(transcript_file)

    if st.session_state.original_review_data is None:
        if original_transcript_file.exists():
            st.session_state.original_review_data = load_transcript(original_transcript_file)
        else:
            st.session_state.original_review_data = []

    if not st.session_state.review_data:
        st.warning("No transcript data found.")
        return False

    sensitive_labels = set(replacements) - {"KEEP", "RESTORE_ORIGINAL", "REMOVE"}

    if not st.session_state.automatic_review_done:
        show_automatic_anonymization_review(
            transcript_file=transcript_file,
            replacements=replacements,
            sensitive_labels=sensitive_labels,
        )
        return False

    return show_full_transcript_review(
        transcript_file=transcript_file,
        replacements=replacements,
        sensitive_labels=sensitive_labels,
    )
