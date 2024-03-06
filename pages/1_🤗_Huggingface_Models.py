import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(layout="centered")
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

def calculate_file_hash(file):
    sha256_hash = hashlib.sha256()
    # Read the file in chunks to avoid using too much memory
    for byte_block in iter(lambda: file.read(4096), b""):
        sha256_hash.update(byte_block)
    file.seek(0)  # Reset file pointer to the beginning
    return sha256_hash.hexdigest()

sample_video_path = "example.mp4"
temp_upload_path = "temp_uploaded_video.mp4"

st.header("Subtitle Generation with Hugging Face LLMs", divider=True)

st.sidebar.caption("""
This app is based on the [Next-gen Kaldi: Generate subtitles for videos](https://huggingface.co/spaces/k2-fsa/generate-subtitles-for-videos) 🤗 Hugging Face Space.

See the [NOTICES](https://github.com/kajarenc/subtitle-demo/blob/main/NOTICES) file for licensing information and credits.
"""
)

language_choices = list(language_to_models.keys())

language_radio = st.radio("Select a language", language_choices, index=0, horizontal=True)

model_selectbox = st.selectbox(
    "Select a model", language_to_models[language_radio], index=0
)

uploaded_video = st.file_uploader("Upload a video to caption", type=["mp4", "webm"])

video_placeholder = st.empty()

if uploaded_video is not None:
    file_hash = calculate_file_hash(uploaded_video)
    video_to_process = uploaded_video
    original_file_name = uploaded_video.name

    # Check if the uploaded file has the same name as the sample video
    if original_file_name == os.path.basename(sample_video_path):
        file_name_to_use = temp_upload_path
    else:
        file_name_to_use = original_file_name
    
    with open(file_name_to_use, "wb") as f:
        f.write(uploaded_video.read())

else:
    st.info("No video uploaded yet. Using the sample video.", icon="📽️")
    # Use the sample video if no video is uploaded
    video_to_process = open(sample_video_path, "rb")
    file_name_to_use = sample_video_path
    file_hash = calculate_file_hash(video_to_process)
    video_to_process.seek(0)

content_change = (st.session_state.get('uploaded_video_hash') != file_hash)
model_change = (st.session_state.get('last_used_model') != model_selectbox)
video_change = (st.session_state.get('uploaded_video_name') != file_name_to_use)

if 'edited_subtitles' not in st.session_state or model_change or video_change or content_change:
    # If the uploaded video is new or different, reset the edited subtitles in the session state
    st.session_state.edited_subtitles = None
    st.session_state.uploaded_video_name = file_name_to_use
    st.session_state.last_used_model = model_selectbox
    st.session_state.uploaded_video_hash = file_hash

video_placeholder.video(video_to_process, start_time=1)

if st.button('Generate Subtitles'):
    st.session_state['generation_triggered'] = True
else:
    st.session_state['generation_triggered'] = False

if 'generation_triggered' in st.session_state and st.session_state['generation_triggered']:
    st.toast("Generating subtitles...", icon="⏳")
    vtt_filename, _, subtitles = process_uploaded_video_file(
        language_radio, model_selectbox, file_name_to_use
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

    edited_subtitles = data_to_webvtt(st.session_state.edited_subtitles.to_dict(orient="records"))
    video_placeholder.video(video_to_process, start_time=1, subtitles=edited_subtitles.encode('utf-8'))

    st.download_button(
        label=f":rainbow[Download {vtt_filename.name}]",
        data=edited_subtitles.encode('utf-8'),
        file_name=f"{vtt_filename.name}",
        mime="text/vtt",
    )

    with st.expander("View raw subtitles"):
        st.text(edited_subtitles)

    # Delete the file from disk if not the sample video.
    if file_name_to_use != sample_video_path:
        os.remove(file_name_to_use)
