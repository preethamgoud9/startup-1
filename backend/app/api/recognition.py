import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.recognition_service import recognition_service

logger = logging.getLogger(__name__)
router = APIRouter()


class CameraStartRequest(BaseModel):
    source: int | str | None = None


class CameraStatusResponse(BaseModel):
    running: bool
    message: str


class RecognitionFrameResponse(BaseModel):
    frame: str | None
    results: list[dict]


@router.post("/camera/start", response_model=CameraStatusResponse)
async def start_camera(request: Request, data: CameraStartRequest):
    try:
        recognition_service.start_camera(source=data.source)
        return CameraStatusResponse(running=True, message="Camera started successfully")
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/camera/stop", response_model=CameraStatusResponse)
async def stop_camera():
    try:
        recognition_service.stop_camera()
        return CameraStatusResponse(running=False, message="Camera stopped successfully")
    except Exception as e:
        logger.error(f"Failed to stop camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live", response_model=RecognitionFrameResponse)
async def get_live_recognition(request: Request):
    face_engine = request.app.state.face_engine

    try:
        results = recognition_service.process_frame(face_engine)
        frame = recognition_service.get_annotated_frame()

        return RecognitionFrameResponse(frame=frame, results=results)
    except Exception as e:
        logger.error(f"Failed to get live recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))
