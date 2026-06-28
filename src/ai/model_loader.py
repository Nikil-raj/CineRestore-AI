from pathlib import Path

import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

from src.ai.registry import ModelRegistry


class ModelLoader:
    """
    Loads AI models from the registry.

    Responsibilities:
    - Read model configuration
    - Build the correct neural network
    - Load weights
    - Return a ready RealESRGANer
    """

    def __init__(self):
        self.registry = ModelRegistry()

    def _build_network(self, model_info):
        """
        Build the correct architecture for the selected model.
        """

        if model_info.weight_file.startswith("realesr-general"):
            return SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=model_info.scale,
                act_type="prelu"
            )

        elif model_info.weight_file == "RealESRGAN_x4plus.pth":
            return RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=model_info.scale
            )

        raise ValueError(
            f"Unsupported model: {model_info.weight_file}"
        )

    def load(self, model_key="general"):

        model_info = self.registry.get(model_key)

        weights_path = self.registry.weights_path(model_key)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found:\n{weights_path}"
            )

        model = self._build_network(model_info)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print("Loading AI Model")
        print("=" * 60)
        print(f"Model  : {model_info.name}")
        print(f"Device : {device.upper()}")
        print(f"Weights: {weights_path}")
        print("=" * 60)

        upsampler = RealESRGANer(
            scale=model_info.scale,
            model_path=str(weights_path),
            model=model,
            tile=model_info.tile_size,
            tile_pad=20,
            pre_pad=0,
            half=False,
            gpu_id=0 if torch.cuda.is_available() else None,
        )

        print("✓ Model loaded successfully")

        return upsampler