"""
CineRestore AI — Model Loader
==============================
Responsible for building and returning a fully-configured RealESRGANer.

Sprint 1.3 changes
------------------
- tile value is now sourced exclusively from
  self.registry.get_tile()  (i.e. ModelRegistry.current_tile_size).
- This allows AIService's retry engine to lower the tile size across
  attempts simply by calling registry.set_retry_tile(attempt) before
  each gpu.reload() / loader.load() cycle — without touching ModelLoader.

Design contract
---------------
- ModelLoader NEVER decides when to reload; that is AIService's job.
- ModelLoader NEVER decides what tile size to use; that is
  ModelRegistry's job via get_tile().
- ModelLoader ALWAYS reads self.registry.get_tile() at load-time so
  that any tile change made by the retry engine is picked up
  automatically on the next reload.
"""

from pathlib import Path

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from src.ai.registry import ModelRegistry, ModelInfo


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Loads AI models from the registry and returns a RealESRGANer.

    Responsibilities
    ----------------
    - Read model configuration from ModelRegistry.
    - Build the correct neural network architecture.
    - Load weights from disk.
    - Instantiate and return a ready-to-use RealESRGANer.

    Non-responsibilities
    --------------------
    - Deciding when to reload (→ AIService).
    - Deciding which tile size to use (→ ModelRegistry.get_tile()).
    - Counting frames or batches (→ AIService / BatchManager).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.registry: ModelRegistry = ModelRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_key: str = "general") -> RealESRGANer:
        """
        Build, configure, and return a RealESRGANer for *model_key*.

        The tile size used is self.registry.get_tile(), which reflects
        whatever the retry engine (AIService) has set via
        registry.set_retry_tile() before calling gpu.reload().

        Parameters
        ----------
        model_key : str
            Key registered in ModelRegistry (e.g. "general").

        Returns
        -------
        RealESRGANer
            Fully initialised upsampler ready for inference.

        Raises
        ------
        FileNotFoundError
            If the weight file does not exist on disk.
        ValueError
            If the model key is unknown or the architecture is
            unsupported.
        NotImplementedError
            If the model key is registered but not yet implemented.
        """

        model_info: ModelInfo = self.registry.get(model_key)
        weights_path: Path = self.registry.weights_path(model_key)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weight file not found:\n{weights_path}\n"
                f"Please download the weights and place them in:\n"
                f"{weights_path.parent}"
            )

        network = self._build_network(model_info)
        tile_size: int = self.registry.get_tile()
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self._print_load_info(model_info, weights_path, tile_size, device)

        upsampler = RealESRGANer(
            scale=model_info.scale,
            model_path=str(weights_path),
            model=network,
            tile=tile_size,
            tile_pad=20,
            pre_pad=0,
            half=False,
            gpu_id=0 if torch.cuda.is_available() else None,
        )

        print("  ✓ Model loaded successfully")
        print("=" * 60)

        return upsampler

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_network(self, model_info: ModelInfo):
        """
        Instantiate the correct neural network for *model_info*.

        Architecture dispatch is based on the weight filename prefix
        and the architecture field in ModelInfo.

        Extension point: add an elif branch here to support future
        architectures (GFPGAN, CodeFormer) without touching load().
        """

        weight_file: str = model_info.weight_file

        # SRVGGNetCompact — realesr-general family
        if weight_file.startswith("realesr-general"):
            return SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=model_info.scale,
                act_type="prelu",
            )

        # RRDBNet — RealESRGAN_x4plus family
        if weight_file == "RealESRGAN_x4plus.pth":
            return RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=model_info.scale,
            )

        raise ValueError(
            f"Unsupported weight file: '{weight_file}'. "
            f"No architecture mapping found."
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _print_load_info(
        model_info: ModelInfo,
        weights_path: Path,
        tile_size: int,
        device: str,
    ) -> None:
        """
        Emit a structured model-load log block to stdout.
        """

        print()
        print("=" * 60)
        print("  Loading AI Model")
        print("=" * 60)
        print(f"  Model   : {model_info.name}")
        print(f"  Device  : {device.upper()}")
        print(f"  Weights : {weights_path.name}")
        print(f"  Tile    : {tile_size} px")
        print(f"  Scale   : {model_info.scale}x")
        print("=" * 60)