#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
from functools import partial
from pathlib import Path

import requests
import torch
import torchaudio
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import GigaAMCTCEventHandler

_LOGGER = logging.getLogger(__name__)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=kwargs["nfilt"],
            window_fn=self.torch_windows[kwargs["window"]],
            mel_scale=mel_scale,
            norm=kwargs["mel_norm"],
            n_fft=kwargs["n_fft"],
            f_max=kwargs.get("highfreq", None),
            f_min=kwargs.get("lowfreq", 0),
            wkwargs=wkwargs,
        )


class AudioToMelSpectrogramPreprocessor(
    NeMoAudioToMelSpectrogramPreprocessor
):  # pylint: disable=too-many-ancestors
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-weights-url",
        help="Path to model weights",
        default="https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/ctc_model_weights.ckpt",
    )
    parser.add_argument(
        "--model-config-url",
        help="Path to model config",
        default="https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/ctc_model_config.yaml",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    # parser.add_argument(
    #     "--language",
    #     help="Default language to set for transcription",
    # )
    # parser.add_argument(
    #     "--compute-type",
    #     default="default",
    #     help="Compute type (float16, int8, etc.)",
    # )
    # parser.add_argument(
    #     "--beam-size",
    #     type=int,
    #     default=5,
    # )
    # parser.add_argument(
    #     "--initial-prompt",
    #     help="Optional text to provide as a prompt for the first window",
    # )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="NeMo",
                description="NVIDIA NeMo framework",
                attribution=Attribution(
                    name="NVIDIA",
                    url="https://github.com/NVIDIA/NeMo/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name="gigaAM-CTC",
                        description="GigaAM (Giga Acoustic Model) - a Conformer-based wav2vec2 foundational model",
                        attribution=Attribution(
                            name="SberDevices",
                            url="https://github.com/salute-developers/GigaAM",
                        ),
                        installed=True,
                        languages=["ru"],
                        version="1",
                    )
                ],
            )
        ],
    )

    # Load model
    model_weights = Path(args.data_dir) / "ctc_model_weights.ckpt"
    if not os.path.isfile(model_weights):
        _LOGGER.debug("Loading %s", args.model_weights_url)
        r = requests.get(args.model_weights_url, timeout=100)
        with open(model_weights, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

    model_config = Path(args.data_dir) / "ctc_model.config.yaml"
    if not os.path.isfile(model_config):
        _LOGGER.debug("Loading %s", args.model_config_url)
        r = requests.get(args.model_config_url, timeout=100)
        with open(model_config, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

    model = EncDecCTCModel.from_config_file(model_config)

    ckpt = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(args.device)
    model.eval()

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            GigaAMCTCEventHandler,
            wyoming_info,
            args,
            model,
            model_lock,
        )
    )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
