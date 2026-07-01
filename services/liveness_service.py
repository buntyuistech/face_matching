import cv2
import numpy as np

from core.constants import Messages


class LivenessService:

    def __init__(self):

        # Future:
        #
        # Load MiniFASNet here
        #
        # self.model = ...

        pass

    # =====================================================
    # Liveness Check
    # =====================================================

    def check(
        self,
        image_path: str,
    ):

        image = cv2.imread(
            image_path,
        )

        if image is None:

            return {
                "passed": False,
                "message": Messages.INTERNAL_ERROR,
            }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # -------------------------------------------------
        # Contrast
        # -------------------------------------------------

        contrast = gray.std()

        if contrast < 20:

            return {
                "passed": False,
                "message": "Poor image contrast.",
            }

        # -------------------------------------------------
        # Texture
        # -------------------------------------------------

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        if laplacian < 50:

            return {
                "passed": False,
                "message": "Low facial texture detected.",
            }

        # -------------------------------------------------
        # Bright Reflection
        # -------------------------------------------------

        bright_pixels = np.sum(
            gray > 250,
        )

        ratio = bright_pixels / gray.size

        if ratio > 0.35:

            return {
                "passed": False,
                "message": "Too much reflection detected.",
            }

        # -------------------------------------------------
        # Passed
        # -------------------------------------------------

        return {
            "passed": True,
            "message": "Liveness passed.",
        }