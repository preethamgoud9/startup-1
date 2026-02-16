import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

from app.models.schemas import (
    StudentEnrollRequest,
    QuickEnrollRequest,
    EnrollmentSessionResponse,
    CaptureImageRequest,
    CaptureImageResponse,
)
from app.services.enrollment_service import enrollment_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/start", response_model=EnrollmentSessionResponse)
async def start_enrollment(request: Request, data: StudentEnrollRequest):
    try:
        session = enrollment_service.start_session(
            student_id=data.student_id,
            name=data.name,
            class_name=data.class_name,
        )

        return EnrollmentSessionResponse(
            session_id=session.session_id,
            student_id=session.student_id,
            required_images=session.required_images,
            current_count=session.images_captured,
            next_pose=session.get_next_pose(),
        )
    except Exception as e:
        logger.error(f"Failed to start enrollment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture", response_model=CaptureImageResponse)
async def capture_image(request: Request, data: CaptureImageRequest):
    face_engine = request.app.state.face_engine

    try:
        success, message, session = enrollment_service.capture_image(
            session_id=data.session_id,
            image_data=data.image_data,
            face_engine=face_engine,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return CaptureImageResponse(
            session_id=session.session_id,
            current_count=session.images_captured,
            required_images=session.required_images,
            next_pose=session.get_next_pose(),
            completed=session.is_complete(),
            message=message,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to capture image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick")
async def quick_enroll(request: Request, data: QuickEnrollRequest):
    """
    Quick enrollment endpoint - enroll a student with just one image.
    Faster but less accurate than full 15-image enrollment.
    """
    face_engine = request.app.state.face_engine

    try:
        success, message = enrollment_service.quick_enroll(
            student_id=data.student_id,
            name=data.name,
            class_name=data.class_name,
            image_data=data.image_data,
            face_engine=face_engine,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {
            "success": True,
            "message": message,
            "student_id": data.student_id,
            "name": data.name,
            "class": data.class_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to quick enroll: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_enroll(
    request: Request,
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(..., alias="class_name"),
    files: List[UploadFile] = File(...),
):
    """
    Enroll a student by uploading photo files (1-15 images).
    Accepts JPEG/PNG files from camera roll, phone, or any source.
    """
    face_engine = request.app.state.face_engine

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 15:
        raise HTTPException(status_code=400, detail="Maximum 15 images allowed")

    try:
        success, message, processed, failed = enrollment_service.upload_enroll(
            student_id=student_id,
            name=name,
            class_name=class_name,
            files=files,
            face_engine=face_engine,
        )

        if not success:
            raise HTTPException(status_code=400, detail=message)

        return {
            "success": True,
            "message": message,
            "student_id": student_id,
            "name": name,
            "class": class_name,
            "images_processed": processed,
            "images_failed": failed,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload enroll: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def cancel_enrollment(session_id: str):
    success = enrollment_service.cancel_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session cancelled successfully"}
