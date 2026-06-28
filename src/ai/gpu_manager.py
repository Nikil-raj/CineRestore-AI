import gc
from typing import Optional

import torch

from src.ai.model_loader import ModelLoader


class GPUManager:
    """
    GPU lifecycle manager.

    Responsibilities
    ----------------
    - Load model
    - Unload model
    - Reload model
    - Clean CUDA memory
    - Report GPU statistics

    GPUManager NEVER decides WHEN to reload.
    AIService owns that responsibility.
    """

    def __init__(
        self,
        model_key="general"
    ):

        self.model_key = model_key

        self.loader = ModelLoader()

        self.upsampler: Optional[object] = None

        self.reload_count = 0

    # --------------------------------------------------

    def load(self):

        if self.upsampler is not None:
            return self.upsampler

        print("\nLoading AI model...\n")

        self.upsampler = self.loader.load(
            self.model_key
        )

        return self.upsampler

    # --------------------------------------------------

    def unload(self):

        if self.upsampler is None:
            return

        print("\nReleasing GPU model...\n")

        del self.upsampler

        self.upsampler = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.ipc_collect()

            torch.cuda.synchronize()

    # --------------------------------------------------

    def reload(self):

        print()

        print("=" * 60)

        print("Reloading AI Model")

        print("=" * 60)

        self.unload()

        self.load()

        self.reload_count += 1

    # --------------------------------------------------

    def cleanup_frame(self):

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.synchronize()

    # --------------------------------------------------

    def get_model(self):

        if self.upsampler is None:

            self.load()

        return self.upsampler

    # --------------------------------------------------

    def stats(self):

        stats = {

            "cuda": torch.cuda.is_available(),

            "reloads": self.reload_count

        }

        if torch.cuda.is_available():

            stats["allocated_mb"] = round(

                torch.cuda.memory_allocated()

                / 1024 ** 2,

                1

            )

            stats["reserved_mb"] = round(

                torch.cuda.memory_reserved()

                / 1024 ** 2,

                1

            )

        return stats