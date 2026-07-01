from enum import Enum


# ==========================================================
# Face Recognition Configuration
# ==========================================================

# InsightFace Model
FACE_MODEL_NAME = "buffalo_l"

# CPU / CUDA Provider
MODEL_PROVIDER = "CPUExecutionProvider"

# Face detector input size
DETECTION_SIZE = (640, 640)

# Number of registration profiles
MIN_REGISTRATION_IMAGES = 3
MAX_REGISTRATION_IMAGES = 4

# Recognition Threshold
#
# NOTE:
# This value will be calibrated after testing.
# Do NOT change it inside the code.
#
MATCH_THRESHOLD = 0.65


# ==========================================================
# Image Quality
# ==========================================================

MIN_FACE_RATIO = 0.22

MIN_BLUR_SCORE = 80.0

MIN_BRIGHTNESS = 45

MAX_BRIGHTNESS = 225


# ==========================================================
# Camera Guidance
# ==========================================================

CENTER_X_TOLERANCE = 0.15

CENTER_Y_TOLERANCE = 0.15


# ==========================================================
# API Messages
# ==========================================================

class Messages:

    FACE_REGISTERED = (
        "Face registered successfully."
    )

    FACE_VERIFIED = (
        "Face verified successfully. Attendance can be marked."
    )

    FACE_NOT_MATCHED = (
        "The captured face does not match your registered face."
    )

    EMPLOYEE_NOT_REGISTERED = (
        "Employee is not registered."
    )

    NO_FACE = (
        "No face detected. Please position your face inside the camera frame."
    )

    MULTIPLE_FACES = (
        "Multiple faces detected. Ensure only one person is visible."
    )

    FACE_TOO_SMALL = (
        "Move closer to the camera."
    )

    FACE_NOT_CENTERED = (
        "Center your face inside the frame."
    )

    IMAGE_BLURRY = (
        "The image is blurry. Hold your phone steady and try again."
    )

    IMAGE_TOO_DARK = (
        "Lighting is too dark. Move to a brighter area."
    )

    IMAGE_TOO_BRIGHT = (
        "Lighting is too bright. Reduce the light and try again."
    )

    LIVENESS_FAILED = (
        "Live face verification failed. Please capture a live selfie."
    )

    INTERNAL_ERROR = (
        "Something went wrong. Please try again."
    )


# ==========================================================
# Verification Result
# ==========================================================

class VerificationStatus(str, Enum):

    VERIFIED = "VERIFIED"

    NOT_MATCHED = "NOT_MATCHED"

    NO_FACE = "NO_FACE"

    MULTIPLE_FACES = "MULTIPLE_FACES"

    LOW_QUALITY = "LOW_QUALITY"

    LIVENESS_FAILED = "LIVENESS_FAILED"

    EMPLOYEE_NOT_REGISTERED = "EMPLOYEE_NOT_REGISTERED"