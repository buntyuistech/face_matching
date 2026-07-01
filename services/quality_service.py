import cv2

from core.constants import (
    CENTER_X_TOLERANCE,
    CENTER_Y_TOLERANCE,
    MAX_BRIGHTNESS,
    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS,
    MIN_FACE_RATIO,
    Messages,
)


class QualityService:

    def check(
        self,
        image,
        face,
    ):

        height, width = image.shape[:2]

        # ======================================================
        # Face Confidence
        # ======================================================

        if face.det_score < 0.60:

            return {
                "passed": False,
                "message": "Face detection confidence is too low.",
            }

        # ======================================================
        # Face Size
        # ======================================================

        x1, y1, x2, y2 = map(
            int,
            face.bbox,
        )

        face_width = x2 - x1
        face_height = y2 - y1

        face_ratio = max(
            face_width / width,
            face_height / height,
        )

        if face_ratio < MIN_FACE_RATIO:

            return {
                "passed": False,
                "message": Messages.FACE_TOO_SMALL,
            }

        # ======================================================
        # Face Center
        # ======================================================

        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        image_center_x = width / 2
        image_center_y = height / 2

        x_offset = abs(
            face_center_x - image_center_x
        ) / width

        y_offset = abs(
            face_center_y - image_center_y
        ) / height

        if (
            x_offset > CENTER_X_TOLERANCE
            or y_offset > CENTER_Y_TOLERANCE
        ):

            return {
                "passed": False,
                "message": Messages.FACE_NOT_CENTERED,
            }

        # ======================================================
        # Blur
        # ======================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        if blur_score < MIN_BLUR_SCORE:

            return {
                "passed": False,
                "message": Messages.IMAGE_BLURRY,
            }

        # ======================================================
        # Brightness
        # ======================================================

        brightness = gray.mean()

        if brightness < MIN_BRIGHTNESS:

            return {
                "passed": False,
                "message": Messages.IMAGE_TOO_DARK,
            }

        if brightness > MAX_BRIGHTNESS:

            return {
                "passed": False,
                "message": Messages.IMAGE_TOO_BRIGHT,
            }

        # ======================================================
        # Passed
        # ======================================================

        return {
            "passed": True,
            "message": "Image quality is good.",
            "details": {
                "faceConfidence": round(
                    float(face.det_score),
                    3,
                ),
                "faceRatio": round(
                    face_ratio,
                    3,
                ),
                "blurScore": round(
                    float(blur_score),
                    2,
                ),
                "brightness": round(
                    float(brightness),
                    2,
                ),
                "centerOffsetX": round(
                    x_offset,
                    3,
                ),
                "centerOffsetY": round(
                    y_offset,
                    3,
                ),
            },
        }