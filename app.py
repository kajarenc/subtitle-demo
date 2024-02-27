import streamlit as st

from assemble_utils import upload_to_assemble, upload_to_s3
from utils import data_to_webvtt, vtt_string_to_dataframe

st.set_page_config(layout="wide")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = dict()

st.header("Generate Subtitles with Assembly AI, and edit them in place!", divider=True)
uploaded_file = st.file_uploader("Choose a file")

left, _, right = st.columns([45, 10, 45])

if uploaded_file is not None:
    if uploaded_file.file_id not in st.session_state.processed_files:
        st.session_state.processed_files = dict()
        st.session_state.processed_files[uploaded_file.file_id] = {
            "file": uploaded_file,
            "status": "pending",
            "public_url": None,
            "vtt": "",
        }
        public_url = upload_to_s3(uploaded_file)
        st.session_state.processed_files[uploaded_file.file_id][
            "public_url"
        ] = public_url
        st.session_state.processed_files[uploaded_file.file_id][
            "status"
        ] = "uploaded_to_s3"

if st.session_state.processed_files:
    for file_id, file_data in st.session_state.processed_files.items():
        if file_data["status"] == "uploaded_to_s3":
            vtt_text = upload_to_assemble(file_data["public_url"])
            st.session_state.processed_files[file_id]["status"] = "processed"
            st.session_state.processed_files[file_id]["vtt"] = vtt_text
        elif file_data["status"] == "transcribed":
            vtt_text = file_data["vtt"]
        with right:
            st.download_button(
                label="Download GENERATED VTT",
                data=st.session_state.processed_files[file_id]["vtt"],
                file_name=f"{file_data['file'].name}.vtt",
                mime="text/vtt",
            )

if item := list(st.session_state.processed_files.values()):
    df = vtt_string_to_dataframe(item[0]["vtt"])
    with left:
        edited_df = st.data_editor(
            df,
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

    edited_webvtt_string = data_to_webvtt(edited_df.to_dict(orient="records"))

    with right:
        st.video(file_data["file"], subtitles=edited_webvtt_string)
        st.download_button(
            label="Download EDITED VTT",
            data=edited_webvtt_string,
            file_name=f"edited.vtt",
            mime="text/vtt",
        )