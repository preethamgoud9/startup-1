from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


class StudentEnrollRequest(BaseModel):
    student_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    class_name: str = Field(..., min_length=1, alias="class")


class QuickEnrollRequest(BaseModel):
    student_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    class_name: str = Field(..., min_length=1, alias="class")
    image_data: str = Field(..., min_length=1)


class EnrollmentSessionResponse(BaseModel):
    session_id: str
    student_id: str
    required_images: int
    current_count: int
    next_pose: Optional[str] = None


class CaptureImageRequest(BaseModel):
    session_id: str
    image_data: str


class CaptureImageResponse(BaseModel):
    session_id: str
    current_count: int
    required_images: int
    next_pose: Optional[str] = None
    completed: bool
    message: str


class RecognitionResult(BaseModel):
    student_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float
    is_known: bool
    timestamp: datetime


class AttendanceRecord(BaseModel):
    student_id: str
    name: str
    class_name: str = Field(..., alias="class")
    date: str
    time: str
    timestamp: datetime


class AttendanceResponse(BaseModel):
    records: list[AttendanceRecord]
    total: int
    date: str
