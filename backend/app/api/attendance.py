import logging
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.models.schemas import AttendanceRecord, AttendanceResponse
from app.services.attendance_service import attendance_service

logger = logging.getLogger(__name__)
router = APIRouter()


class MarkAttendanceRequest(BaseModel):
    student_id: str
    name: str
    class_name: str
    confidence: float


class MarkAttendanceResponse(BaseModel):
    success: bool
    message: str


class ExportRequest(BaseModel):
    date: Optional[str] = None
    class_filter: Optional[str] = None


@router.get("/today", response_model=AttendanceResponse)
async def get_today_attendance():
    try:
        records = attendance_service.get_today_attendance()
        return AttendanceResponse(
            records=[AttendanceRecord(**r) for r in records],
            total=len(records),
            date=date.today().isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to get today's attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/date/{target_date}", response_model=AttendanceResponse)
async def get_attendance_by_date(target_date: str):
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        records = attendance_service.get_attendance_by_date(target)
        return AttendanceResponse(
            records=[AttendanceRecord(**r) for r in records],
            total=len(records),
            date=target_date,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Failed to get attendance for {target_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark", response_model=MarkAttendanceResponse)
async def mark_attendance(data: MarkAttendanceRequest):
    try:
        success, message = attendance_service.mark_attendance(
            student_id=data.student_id,
            name=data.name,
            class_name=data.class_name,
            confidence=data.confidence,
        )
        return MarkAttendanceResponse(success=success, message=message)
    except Exception as e:
        logger.error(f"Failed to mark attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_attendance(data: ExportRequest):
    try:
        target_date = None
        if data.date:
            target_date = datetime.strptime(data.date, "%Y-%m-%d").date()

        export_path = attendance_service.export_to_excel(
            target_date=target_date,
            class_filter=data.class_filter,
        )

        return FileResponse(
            path=str(export_path),
            filename=export_path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to export attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
