import json
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

from app.core.config import settings

SETTINGS_FILE = Path("data/settings.json")

class CameraSettingsUpdate(BaseModel):
    source_type: str  # "usb" or "rtsp"
    rtsp_url: Optional[str] = ""
    usb_device_id: Optional[int] = 0

def load_dynamic_settings():
    if not SETTINGS_FILE.exists():
        return {
            "source_type": settings.camera.source_type,
            "rtsp_url": settings.camera.rtsp_url,
            "usb_device_id": settings.camera.usb_device_id
        }
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

def save_dynamic_settings(new_settings: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(new_settings, f, indent=4)
    
    # Update runtime settings
    settings.camera.source_type = new_settings.get("source_type", settings.camera.source_type)
    settings.camera.rtsp_url = new_settings.get("rtsp_url", settings.camera.rtsp_url)
    settings.camera.usb_device_id = new_settings.get("usb_device_id", settings.camera.usb_device_id)
