import numpy as np

from src.ai.gpu_manager import GPUManager


class Upscaler:
    """
    Wrapper around RealESRGAN.

    Responsibilities
    ----------------
    - Hold the current model
    - Enhance one image
    - Release/update model references

    NEVER loads or reloads models.
    """

    def __init__(self, gpu_manager: GPUManager):

        self.gpu = gpu_manager

        self.model = self.gpu.get_model()

    # --------------------------------------------------

    def update_model(self):
        """
        Called after GPUManager.reload()
        """

        self.model = self.gpu.get_model()

    # --------------------------------------------------

    def release_model(self):
        """
        Release our reference before GPU reload.
        """

        self.model = None

    # --------------------------------------------------

    def upscale(
        self,
        image: np.ndarray,
        outscale: int = 4
    ) -> np.ndarray:

        if self.model is None:

            self.update_model()

        result, _ = self.model.enhance(
            image,
            outscale=outscale
        )

        return result