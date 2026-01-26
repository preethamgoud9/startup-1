from fastapi import APIRouter, HTTPException, Depends
from app.core.settings_manager import load_dynamic_settings, save_dynamic_settings, CameraSettingsUpdate
from app.api.auth import get_current_user
from typing import Annotated
import cv2

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("/camera")
async def get_camera_settings(_user: Annotated[dict, Depends(get_current_user)]):
    return load_dynamic_settings()

@router.post("/camera")
async def update_camera_settings(
    data: CameraSettingsUpdate,
    _user: Annotated[dict, Depends(get_current_user)]
):
    # Validate source
    source = data.rtsp_url if data.source_type == "rtsp" else data.usb_device_id
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=400, 
            detail=f"Unable to connect to camera source: {source}"
        )
    cap.release()
    
    save_dynamic_settings(data.model_dump())
    return {"status": "success", "message": "Camera configuration updated"}
