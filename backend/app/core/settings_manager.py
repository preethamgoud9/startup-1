import json
from pathlib import Path

from typing import Optional

from pydantic import BaseModel

from app.core.config import settings

SETTINGS_FILE = Path("data/settings.json")

class CameraSettingsUpdate(BaseModel):
    source_type: str = "usb"  # "usb" or "rtsp"
    usb_device_id: int = 0
    rtsp_url: Optional[str] = ""

def load_dynamic_settings():
    if not SETTINGS_FILE.exists():
        return {
            "source_type": settings.camera.source_type,
            "usb_device_id": settings.camera.usb_device_id,
            "rtsp_url": settings.camera.rtsp_url,
        }
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


def _update_runtime_camera_settings(new_settings: dict):
    if not isinstance(new_settings, dict):
        return

    settings.camera.source_type = new_settings.get("source_type", settings.camera.source_type)
    settings.camera.usb_device_id = new_settings.get("usb_device_id", settings.camera.usb_device_id)
    settings.camera.rtsp_url = new_settings.get("rtsp_url", settings.camera.rtsp_url)

def save_dynamic_settings(new_settings: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(new_settings, f, indent=4)
    
    _update_runtime_camera_settings(new_settings)


def sync_runtime_camera_settings():
    """Load persisted camera settings (if any) and apply them at runtime."""
    _update_runtime_camera_settings(load_dynamic_settings())
