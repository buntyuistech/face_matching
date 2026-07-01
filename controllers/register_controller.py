from core.constants import (
    MAX_REGISTRATION_IMAGES,
    Messages,
    MIN_REGISTRATION_IMAGES,
)
from services.shared import (
    face_service,
    storage_service,
)


class RegisterController:

    def register(
        self,
        employee_id: str,
        image_path: str,
        registration_index: int,
        has_glasses_profile: bool = False,
    ):

        # ======================================================
        # Validate Registration Index
        # ======================================================

        if registration_index < 1 or registration_index > MAX_REGISTRATION_IMAGES:
            raise Exception("Invalid registration index.")

        # ======================================================
        # Create Employee Folder
        # ======================================================

        if registration_index == 1:

            if storage_service.employee_exists(
                employee_id,
            ):
                storage_service.delete_employee(
                    employee_id,
                )

            storage_service.create_employee(
                employee_id,
            )

        # ======================================================
        # Generate Face Embedding
        # ======================================================

        embedding = face_service.extract_embedding_from_path(
            image_path,
        )

        # ======================================================
        # Save Profile Image
        # ======================================================

        storage_service.save_profile_image(
            employee_id=employee_id,
            image_path=image_path,
            index=registration_index,
        )

        # ======================================================
        # Save Embedding
        # ======================================================

        storage_service.save_embedding(
            employee_id=employee_id,
            embedding=embedding,
            index=registration_index,
        )

        # ======================================================
        # Registration Completed?
        # ======================================================

        registration_completed = (
            registration_index == MIN_REGISTRATION_IMAGES
            and not has_glasses_profile
        ) or (
            registration_index == MAX_REGISTRATION_IMAGES
        )

        # ======================================================
        # Save Metadata
        # ======================================================

        storage_service.save_metadata(
            employee_id=employee_id,
            total_profiles=registration_index,
            has_glasses_profile=has_glasses_profile,
        )

        # ======================================================
        # Response
        # ======================================================

        if registration_completed:

            message = Messages.FACE_REGISTERED

        else:

            message = (
                f"Face profile "
                f"{registration_index} "
                f"registered successfully."
            )

        return {
            "success": True,
            "employeeId": employee_id,
            "registrationIndex": registration_index,
            "registrationCompleted": registration_completed,
            "message": message,
        }