from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    """
    Describes one AI model.
    """

    name: str
    weight_file: str
    scale: int
    architecture: str
    tile_size: int
    description: str


class ModelRegistry:
    """
    Registry of all AI models supported by CineRestore AI.
    """

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        self.weights_dir = (
            self.project_root
            / "models"
            / "RealESRGAN"
            / "weights"
        )

        self.models = {

            "general": ModelInfo(
                name="RealESRGAN General",
                weight_file="realesr-general-x4v3.pth",
                scale=4,
                architecture="RRDBNet",
                tile_size=200,
                description="General purpose movie upscaling."
            ),

            "general_denoise": ModelInfo(
                name="RealESRGAN General Denoise",
                weight_file="realesr-general-wdn-x4v3.pth",
                scale=4,
                architecture="RRDBNet",
                tile_size=200,
                description="General model with stronger denoising."
            ),

            # Future models

            "anime": None,

            "gfpgan": None,

            "codeformer": None,

            "rife": None,
        }

    def get(self, key: str) -> ModelInfo:

        if key not in self.models:

            raise ValueError(
                f"Unknown model '{key}'"
            )

        model = self.models[key]

        if model is None:

            raise NotImplementedError(
                f"Model '{key}' has not been implemented yet."
            )

        return model

    def weights_path(self, key: str) -> Path:

        model = self.get(key)

        return self.weights_dir / model.weight_file

    def list_models(self):

        return [
            k
            for k, v in self.models.items()
            if v is not None
        ]