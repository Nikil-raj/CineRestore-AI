from pathlib import Path

import cv2

from src.ai.gpu_manager import GPUManager
from src.ai.upscaler import Upscaler


class AIEngine:
    """
    AIEngine enhances ONE frame.

    It does not:
    - reload models
    - count frames
    - process batches
    """

    def __init__(self, gpu_manager: GPUManager):

        self.gpu = gpu_manager

        self.upscaler = Upscaler(
            gpu_manager
        )

    # --------------------------------------------------

    def update_model(self):
        """
        Called by AIService after GPU reload.
        """

        self.upscaler.update_model()

    # --------------------------------------------------

    def release_model(self):
        """
        Called before GPU reload.
        """

        self.upscaler.release_model()

    # --------------------------------------------------

    def enhance_frame(
        self,
        input_path,
        output_path,
        outscale=4
    ):

        input_path = Path(input_path)
        output_path = Path(output_path)

        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image:\n{input_path}"
            )

        result = self.upscaler.upscale(
            image,
            outscale
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            result
        )

        return output_path

    # --------------------------------------------------

    def enhance_image(
        self,
        input_path,
        output_path,
        outscale=4
    ):
        """
        Backward compatibility.
        """

        return self.enhance_frame(
            input_path,
            output_path,
            outscale
        )