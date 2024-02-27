import logging
import subprocess
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import sherpa_onnx
from model import sample_rate


@dataclass
class Segment:
    start: float
    duration: float
    text: str = ""

    @property
    def end(self):
        return self.start + self.duration

    def __str__(self):
        s = f"0{timedelta(seconds=self.start)}"[:-3]
        s += " --> "
        s += f"0{timedelta(seconds=self.end)}"[:-3]
        s = s.replace(".", ",")
        s += "\n"
        s += self.text
        return s


def decode(
    recognizer: sherpa_onnx.OfflineRecognizer,
    vad: sherpa_onnx.VoiceActivityDetector,
    filename: str,
) -> str:
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        filename,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]

    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    frames_per_read = int(sample_rate * 100)  # 100 second

    window_size = 512

    buffer = []

    segment_list = []

    logging.info("Started!")

    while True:
        # *2 because int16_t has two bytes
        data = process.stdout.read(frames_per_read * 2)
        if not data:
            break

        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768

        buffer = np.concatenate([buffer, samples])
        while len(buffer) > window_size:
            vad.accept_waveform(buffer[:window_size])
            buffer = buffer[window_size:]

        streams = []
        segments = []
        while not vad.empty():
            segment = Segment(
                start=vad.front.start / sample_rate,
                duration=len(vad.front.samples) / sample_rate,
            )
            segments.append(segment)

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, vad.front.samples)

            streams.append(stream)

            vad.pop()

        for s in streams:
            recognizer.decode_stream(s)

        for seg, stream in zip(segments, streams):
            seg.text = stream.result.text.strip()
            segment_list.append(seg)

    return "\n\n".join(f"{i}\n{seg}" for i, seg in enumerate(segment_list, 1))