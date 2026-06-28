"""
CineRestore AI — Model Registry
================================
Central catalogue of all AI models supported by CineRestore AI.

Sprint 1.3 additions
--------------------
- ModelRegistry now owns the adaptive tile state used by the retry engine.
- current_tile_size  : the tile value that ModelLoader will use on the
                       next load() call.
- tile_sequence      : ordered list of tile sizes tried across retries.
- set_retry_tile()   : advance tile state to a specific retry attempt.
- reset_tile()       : restore tile state to model default.
- get_tile()         : read current effective tile size.

Design Notes
------------
- ModelInfo remains a frozen dataclass; tile_size on ModelInfo is the
  *default* tile for that model, never mutated.
- All mutable tile state lives on ModelRegistry, not on ModelInfo.
- ModelLoader reads self.registry.get_tile() so it always picks up
  the correct tile without any caller changes.
- Future models (GFPGAN, CodeFormer, RIFE) are pre-registered as None
  placeholders; implement them without changing callers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Tile sequence for adaptive retry
# ---------------------------------------------------------------------------

#: Tile sizes tried on attempt 1, 2, 3 respectively.
#: Extend this list to add more fallback steps without changing any caller.
DEFAULT_TILE_SEQUENCE: List[int] = [128, 96, 64]


# ---------------------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    """
    Immutable descriptor for one AI model.

    Fields
    ------
    name          : Human-readable display name.
    weight_file   : Filename inside the weights directory.
    scale         : Upscale factor (e.g. 4).
    architecture  : Neural network class name ("SRVGGNetCompact" / "RRDBNet").
    tile_size     : Default tile size for this model (used to seed
                    ModelRegistry.reset_tile()).
    description   : Short human-readable description.
    """

    name: str
    weight_file: str
    scale: int
    architecture: str
    tile_size: int
    description: str


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Registry of all AI models supported by CineRestore AI.

    Adaptive tile API (Sprint 1.3)
    ------------------------------
    set_retry_tile(attempt)  — set current tile to tile_sequence[attempt-1]
    reset_tile(model_key)    — restore current tile to model's default
    get_tile()               — return current effective tile size
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:

        self.project_root: Path = Path(__file__).resolve().parents[2]

        self.weights_dir: Path = (
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
                architecture="SRVGGNetCompact",
                tile_size=128,
                description="General purpose movie upscaling.",
            ),

            "general_denoise": ModelInfo(
                name="RealESRGAN General Denoise",
                weight_file="realesr-general-wdn-x4v3.pth",
                scale=4,
                architecture="SRVGGNetCompact",
                tile_size=128,
                description="General model with stronger denoising.",
            ),

            # ── Future models ─────────────────────────────────────────
            # Register here when implemented; callers do not change.

            "anime": None,       # RealESRGAN anime variant

            "gfpgan": None,      # Face restoration

            "codeformer": None,  # Face restoration (transformer)

            "rife": None,        # Frame interpolation
        }

        # Adaptive tile state — mutable, owned exclusively by the registry.
        self.tile_sequence: List[int] = list(DEFAULT_TILE_SEQUENCE)
        self.current_tile_size: int = self.tile_sequence[0]

    # ------------------------------------------------------------------
    # Model lookup
    # ------------------------------------------------------------------

    def get(self, key: str) -> ModelInfo:
        """
        Return the ModelInfo for *key*.

        Raises
        ------
        ValueError          : key is not registered.
        NotImplementedError : key is registered but not yet implemented.
        """

        if key not in self.models:
            raise ValueError(
                f"Unknown model key '{key}'. "
                f"Available: {self.list_models()}"
            )

        model = self.models[key]

        if model is None:
            raise NotImplementedError(
                f"Model '{key}' is registered but has not been "
                f"implemented yet."
            )

        return model

    # ------------------------------------------------------------------

    def weights_path(self, key: str) -> Path:
        """Return the absolute path to the weight file for *key*."""

        model = self.get(key)
        return self.weights_dir / model.weight_file

    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return a list of all implemented (non-None) model keys."""

        return [k for k, v in self.models.items() if v is not None]

    # ------------------------------------------------------------------
    # Adaptive tile API (Sprint 1.3)
    # ------------------------------------------------------------------

    def get_tile(self) -> int:
        """
        Return the tile size that ModelLoader should use on the next
        load() call.
        """

        return self.current_tile_size

    # ------------------------------------------------------------------

    def set_retry_tile(self, attempt: int) -> None:
        """
        Advance current_tile_size to the value corresponding to
        *attempt* (1-indexed).

        If *attempt* exceeds the length of tile_sequence, the last
        (smallest) tile in the sequence is used, ensuring the caller
        never receives an IndexError.

        Parameters
        ----------
        attempt : int
            Retry attempt number, starting at 1.
        """

        index = min(attempt - 1, len(self.tile_sequence) - 1)
        self.current_tile_size = self.tile_sequence[index]

    # ------------------------------------------------------------------

    def reset_tile(self, model_key: Optional[str] = None) -> None:
        """
        Reset current_tile_size to the model's default tile size.

        If *model_key* is None, resets to the first value in
        tile_sequence (i.e. DEFAULT_TILE_SEQUENCE[0]).

        Parameters
        ----------
        model_key : str, optional
            When supplied, seed from ModelInfo.tile_size for that model.
        """

        if model_key is not None:
            model = self.get(model_key)
            self.current_tile_size = model.tile_size
        else:
            self.current_tile_size = self.tile_sequence[0]