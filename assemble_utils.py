import assemblyai as aai
import streamlit as st

aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

transcriber = aai.Transcriber()


@st.cache_data
def upload_to_assemle(file_url):
    transcript = transcriber.transcribe(file_url)
    subtitles = transcript.export_subtitles_vtt()

    return subtitles
