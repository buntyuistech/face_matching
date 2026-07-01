from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from controllers.verify_controller import VerifyController

router = APIRouter(
    prefix="/verify",
    tags=["Verify"],
)

controller = VerifyController()


@router.post("")
async def verify(
    employee_id: str = Form(...),
    image: UploadFile = File(...),
):
    temp_file = None

    try:

        suffix = Path(image.filename).suffix or ".jpg"

        with NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as file:

            file.write(await image.read())
            temp_file = file.name

        result = controller.verify(
            employee_id=employee_id,
            image_path=temp_file,
        )

        return result

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