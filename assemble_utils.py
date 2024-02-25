import assemblyai as aai
import boto3
import streamlit as st

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


aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
transcriber = aai.Transcriber()


@st.cache_data
def upload_to_assemble(file_url):
    transcript = transcriber.transcribe(file_url)
    subtitles = transcript.export_subtitles_vtt()

    return subtitles
