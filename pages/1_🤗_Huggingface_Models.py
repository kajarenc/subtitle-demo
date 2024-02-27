import logging
import os
from pathlib import Path

import streamlit as st
from streamlit.elements.lib.subtitle_utils import _srt_to_vtt

from decode import decode
from model import get_pretrained_model, get_vad, language_to_models
from utils import data_to_webvtt, vtt_string_to_dataframe


def show_file_info(in_filename: str):
    logging.info(f"Input file: {in_filename}")
    _ = os.system(f"ffprobe -hide_banner -i '{in_filename}'")


def process(language: str, repo_id: str, in_filename: str):
    recognizer = get_pretrained_model(repo_id)
    vad = get_vad()

    result = decode(recognizer, vad, in_filename)
    logging.info(result)

    vtt_filename = Path(in_filename).with_suffix(".vtt")
    result = _srt_to_vtt(result)

    show_file_info(in_filename)
    logging.info("Done")

    return (
        vtt_filename,
        st.toast("Success! Download the subtitles below.", icon="🍿"),
        result,
    )

def process_uploaded_video_file(
    language: str,
    repo_id: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return (
            "",
            st.error(
                "Please first upload a file"
            ),
            "",
            "",
        )

    logging.info(f"Processing uploaded file: {in_filename}")

    ans = process(language, repo_id, in_filename)
    return ans[0], ans[1], ans[2]


st.header("Subtitle Generation with Hugging Face LLMs", divider=True)

language_choices = list(language_to_models.keys())

language_radio = st.radio("Select a language", language_choices, index=0, horizontal=True)

model_selectbox = st.selectbox(
    "Select a model", language_to_models[language_radio], index=0
)

uploaded_video = st.file_uploader("Upload a video from disk", type=["mp4", "webm"])

video_placeholder = st.empty()

if uploaded_video is not None:
    if 'edited_subtitles' not in st.session_state or st.session_state.uploaded_video_name != uploaded_video.name:
        # If the uploaded video is new or different, reset the edited subtitles in the session state
        st.session_state.edited_subtitles = None
        st.session_state.uploaded_video_name = uploaded_video.name
    
    with open(uploaded_video.name, "wb") as f:
        f.write(uploaded_video.read())

    video_placeholder.video(uploaded_video, start_time=1)

    st.toast("Generating subtitles...", icon="⏳")
    vtt_filename, _, subtitles = process_uploaded_video_file(
        language_radio, model_selectbox, uploaded_video.name
    )

    # Determine the initial data for the data editor
    initial_data = st.session_state.edited_subtitles if st.session_state.edited_subtitles is not None else vtt_string_to_dataframe(subtitles.decode("utf-8"))

    edited_df = st.data_editor(
        initial_data,
        use_container_width=True,
        column_config={
            "text": st.column_config.TextColumn(
                "Subtitle text",
                help="The subtitle text to be displayed from the start time to the end time. 🎈",
            ),
            "start": st.column_config.TimeColumn(
                "Start time",
                help="The start time of the subtitle. 🕒",
            ),
            "end": st.column_config.TimeColumn(
                "End time",
                help="The end time of the subtitle. 🕒",
            ),
        },
    )
    
    # Store the edited subtitles in session_state
    st.session_state.edited_subtitles = edited_df
    
    edited_subtitles = data_to_webvtt(st.session_state.edited_subtitles)
    video_placeholder.video(uploaded_video, start_time=1, subtitles=edited_subtitles)

    st.download_button(
        label=f":rainbow[Download {vtt_filename.name}]",
        data=edited_subtitles.encode('utf-8'),
        file_name=f"{vtt_filename.name}",
        mime="text/vtt",
    )
    
    with st.expander("View raw subtitles"):
        st.text(edited_subtitles)

    # Delete the file from disk
    os.remove(uploaded_video.name)
