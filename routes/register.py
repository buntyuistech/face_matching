from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from controllers.register_controller import RegisterController

router = APIRouter(
    prefix="/register",
    tags=["Register"],
)

controller = RegisterController()


@router.post("")
async def register(
    employee_id: str = Form(...),
    registration_index: int = Form(...),
    has_glasses_profile: bool = Form(False),
    image: UploadFile = File(...),
):
    temp_file = None

    try:

        if registration_index not in [1, 2, 3, 4]:
            raise HTTPException(
                status_code=400,
                detail="registration_index must be between 1 and 4.",
            )

        suffix = Path(image.filename).suffix or ".jpg"

        with NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as file:
            file.write(await image.read())
            temp_file = file.name

        result = controller.register(
            employee_id=employee_id,
            image_path=temp_file,
            registration_index=registration_index,
            has_glasses_profile=has_glasses_profile,
        )

        return {
            "success": True,
            "message": result["message"],
            "data": result,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    finally:
        if temp_file and Path(temp_file).exists():
            Path(temp_file).unlink()