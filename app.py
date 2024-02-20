import streamlit as st
import boto3

from assemble_utils import upload_to_assemle


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

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    if uploaded_file.file_id not in st.session_state.processed_files:
        st.session_state.processed_files = dict()
        st.write("VIDEO WITHOUT SUBTITLES")
        st.video(uploaded_file)
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
    st.write("Uploaded files:")
    for file_id, file_data in st.session_state.processed_files.items():
        if file_data["status"] == "uploaded_to_s3":
            st.write(f"Public URL: {file_data['public_url']}")
            vtt_text = upload_to_assemle(file_data["public_url"])
            st.download_button(
                label="Download VTT",
                data=vtt_text,
                file_name=f"{file_data['file'].name}.vtt",
                mime="text/vtt",
            )

        st.session_state.processed_files[file_id]["status"] = "processed"
        st.session_state.processed_files[file_id]["vtt"] = vtt_text
        st.write("VIDEO WITH SUBTITLES")
        st.video(file_data["file"], subtitles=file_data["vtt"])


st.write("-----------------")
st.write(st.session_state.processed_files)
