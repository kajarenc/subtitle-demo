from functools import lru_cache

import sherpa_onnx
import streamlit as st
from huggingface_hub import hf_hub_download

sample_rate = 16000


def _get_nn_model_filename(
    repo_id: str,
    filename: str,
    subfolder: str = "exp",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


get_file = _get_nn_model_filename


def _get_bpe_model_filename(
    repo_id: str,
    filename: str = "bpe.model",
    subfolder: str = "data/lang_bpe_500",
) -> str:
    bpe_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return bpe_model_filename


def _get_token_filename(
    repo_id: str,
    filename: str = "tokens.txt",
    subfolder: str = "data/lang_char",
) -> str:
    token_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return token_filename


@st.cache_resource(max_entries=10)
def _get_whisper_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    name = repo_id.split("-")[1]
    assert name in ("tiny.en", "base.en", "small.en", "medium.en"), repo_id
    full_repo_id = "csukuangfj/sherpa-onnx-whisper-" + name
    encoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"{name}-encoder.int8.onnx",
        subfolder=".",
    )

    decoder = _get_nn_model_filename(
        repo_id=full_repo_id,
        filename=f"{name}-decoder.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(
        repo_id=full_repo_id, subfolder=".", filename=f"{name}-tokens.txt"
    )

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        num_threads=2,
    )

    return recognizer


@st.cache_resource(max_entries=10)
def _get_paraformer_zh_pre_trained_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.int8.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    return recognizer


@st.cache_resource(max_entries=10)
def _get_russian_pre_trained_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in (
        "alphacep/vosk-model-ru",
        "alphacep/vosk-model-small-ru",
    ), repo_id

    if repo_id == "alphacep/vosk-model-ru":
        model_dir = "am-onnx"
    elif repo_id == "alphacep/vosk-model-small-ru":
        model_dir = "am"

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder.onnx",
        subfolder=model_dir,
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder.onnx",
        subfolder=model_dir,
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner.onnx",
        subfolder=model_dir,
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="lang")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )

    return recognizer


@lru_cache(maxsize=2)
def get_vad() -> sherpa_onnx.VoiceActivityDetector:
    vad_model = _get_nn_model_filename(
        repo_id="csukuangfj/vad",
        filename="silero_vad.onnx",
        subfolder=".",
    )

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = vad_model
    config.silero_vad.min_silence_duration = 0.15
    config.silero_vad.min_speech_duration = 0.25
    config.sample_rate = sample_rate

    vad = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=180,
    )

    return vad


@st.cache_resource(max_entries=10)
def get_pretrained_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    if repo_id in chinese_models:
        return chinese_models[repo_id](repo_id)
    elif repo_id in english_models:
        return english_models[repo_id](repo_id)
    elif repo_id in chinese_english_mixed_models:
        return chinese_english_mixed_models[repo_id](repo_id)
    elif repo_id in russian_models:
        return russian_models[repo_id](repo_id)
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


def _get_wenetspeech_pre_trained_model(repo_id):
    assert repo_id in (
        "csukuangfj/sherpa-onnx-conformer-zh-stateless2-2023-05-23",
    ), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-99-avg-1.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )

    return recognizer


def _get_multi_zh_hans_pre_trained_model(repo_id):
    assert repo_id in ("zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-20-avg-1.onnx",
        subfolder=".",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-20-avg-1.onnx",
        subfolder=".",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-20-avg-1.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )

    return recognizer


def _get_english_model(repo_id: str) -> sherpa_onnx.OfflineRecognizer:
    assert (
        repo_id
        == "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
    ), repo_id

    encoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="encoder-epoch-30-avg-4.onnx",
        subfolder="exp",
    )

    decoder_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="decoder-epoch-30-avg-4.onnx",
        subfolder="exp",
    )

    joiner_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="joiner-epoch-30-avg-4.onnx",
        subfolder="exp",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder="lang_bpe_500")

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder_model,
        decoder=decoder_model,
        joiner=joiner_model,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )

    return recognizer


chinese_models = {
    "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28": _get_paraformer_zh_pre_trained_model,
    "csukuangfj/sherpa-onnx-conformer-zh-stateless2-2023-05-23": _get_wenetspeech_pre_trained_model,  # noqa
    "zrjin/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2": _get_multi_zh_hans_pre_trained_model,  # noqa
}

english_models = {
    "whisper-tiny.en": _get_whisper_model,
    "whisper-base.en": _get_whisper_model,
    "whisper-small.en": _get_whisper_model,
    "whisper-distil-small.en": _get_whisper_model,
    "whisper-medium.en": _get_whisper_model,
    "whisper-distil-medium.en": _get_whisper_model,
    "yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04": _get_english_model,  # noqa
}

chinese_english_mixed_models = {
    "csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28": _get_paraformer_zh_pre_trained_model,
}

russian_models = {
    "alphacep/vosk-model-ru": _get_russian_pre_trained_model,
    "alphacep/vosk-model-small-ru": _get_russian_pre_trained_model,
}

language_to_models = {
    "English": list(english_models.keys()),
    "Chinese": list(chinese_models.keys()),
    "Chinese+English": list(chinese_english_mixed_models.keys()),
    "Russian": list(russian_models.keys()),
}

