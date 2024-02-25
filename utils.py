import pandas as pd
from datetime import datetime, time
import webvtt
import pandas as pd
import streamlit as st
import io


def time_to_webvtt_timestamp(t: time):
    """Convert a datetime.time object to a WebVTT timestamp string."""
    return f"{t.strftime('%H:%M:%S')}.000"


def string_to_time(s: str):
    """Convert a string to a datetime.time object."""
    return datetime.strptime(s, "%H:%M:%S.%f").time()


def vtt_string_to_dataframe(vtt_string: str) -> pd.DataFrame:
    time_epsilon = pd.Timedelta("00:00:00.1")

    buffer = io.StringIO(vtt_string)

    vtt = webvtt.read_buffer(buffer=buffer)

    df = pd.DataFrame(
        [
            [
                pd.to_datetime(v.start),
                pd.to_datetime(v.end),
                v.text.splitlines()[-1],
            ]
            for v in vtt
        ],
        columns=["start", "end", "text"],
    )
    df = df.where(df.end - df.start > time_epsilon).dropna()
    df["start"] = df["start"].apply(time_to_webvtt_timestamp)
    df["end"] = df["end"].apply(time_to_webvtt_timestamp)
    df["start"] = df["start"].apply(string_to_time)
    df["end"] = df["end"].apply(string_to_time)
    return df


def data_to_webvtt(data) -> str:
    webvtt_content = "WEBVTT\n\n"

    for index, entry in enumerate(data, start=1):
        start_time = time_to_webvtt_timestamp(entry["start"])
        end_time = time_to_webvtt_timestamp(entry["end"])
        text = entry["text"].replace("\n", " ")

        webvtt_content += f"{index}\n{start_time} --> {end_time}\n{text}\n\n"

    return webvtt_content
