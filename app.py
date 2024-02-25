import streamlit as st
import boto3

from assemble_utils import upload_to_assemle
from utils import vtt_string_to_dataframe, data_to_webvtt

st.set_page_config(layout="wide")

AWS_S3_BUCKET_NAME = st.secrets["AWS_S3_BUCKET_NAME"]
AWS_REGION = st.secrets["AWS_REGION"]
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]


@st.cache_data
def upload_to_s3(uploaded_file):
    s3_client = boto3.client(
        service_name="s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    s3_client.upload_fileobj(uploaded_file, AWS_S3_BUCKET_NAME, uploaded_file.name)

    url = "https://s3-%s.amazonaws.com/%s/%s" % (
        AWS_REGION,
        AWS_S3_BUCKET_NAME,
        uploaded_file.name,
    )
    return url


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
            vtt_text = upload_to_assemle(file_data["public_url"])
            with right:
                st.download_button(
                    label="Download GENERATED VTT",
                    data=vtt_text,
                    file_name=f"{file_data['file'].name}.vtt",
                    mime="text/vtt",
                )
            st.session_state.processed_files[file_id]["status"] = "processed"
            st.session_state.processed_files[file_id]["vtt"] = vtt_text
        elif file_data["status"] == "transcribed":
            vtt_text = file_data["vtt"]


if item := list(st.session_state.processed_files.values()):
    vtt = item[0]["vtt"]
    df = vtt_string_to_dataframe(vtt)
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
