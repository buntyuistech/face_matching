from core.constants import Messages
from services.shared import (
    face_service,
    liveness_service,
    quality_service,
    storage_service,
)


class VerifyController:

    def verify(
        self,
        employee_id: str,
        image_path: str,
    ):

        # =====================================================
        # Employee Exists
        # =====================================================

        if not storage_service.employee_exists(
            employee_id,
        ):
            return {
                "success": False,
                "message": Messages.EMPLOYEE_NOT_REGISTERED,
            }

        # =====================================================
        # Load Image
        # =====================================================

        image = face_service.load_image(
            image_path,
        )

        # =====================================================
        # Detect Face
        # =====================================================

        face = face_service.detect_face(
            image,
        )

        # =====================================================
        # Image Quality
        # =====================================================

        quality = quality_service.check(
            image=image,
            face=face,
        )

        if not quality["passed"]:

            return {
                "success": False,
                "message": quality["message"],
            }

        # =====================================================
        # Liveness
        # =====================================================

        liveness = liveness_service.check(
            image_path,
        )

        if not liveness["passed"]:

            return {
                "success": False,
                "message": liveness["message"],
            }

        # =====================================================
        # Current Embedding
        # =====================================================

        current_embedding = face_service.extract_embedding(
            face,
        )

        # =====================================================
        # Registered Embeddings
        # =====================================================

        registered_embeddings = storage_service.load_embeddings(
            employee_id,
        )

        if len(registered_embeddings) == 0:

            return {
                "success": False,
                "message": "No registered face found.",
            }

        # =====================================================
        # Face Matching
        # =====================================================

        result = face_service.verify_face(
            current_embedding=current_embedding,
            registered_embeddings=registered_embeddings,
        )

        # =====================================================
        # Response
        # =====================================================

        if result["matched"]:

            return {
                "success": True,
                "message": Messages.FACE_VERIFIED,
                "data": {
                    "employeeId": employee_id,
                    "similarity": result["similarity"],
                },
            }

        return {
            "success": False,
            "message": Messages.FACE_NOT_MATCHED,
            "data": {
                "employeeId": employee_id,
                "similarity": result["similarity"],
            },
        }