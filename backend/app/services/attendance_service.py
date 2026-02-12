import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class AttendanceService:
    def __init__(self):
        self.attendance_dir = Path(settings.attendance.data_dir)
        self.attendance_dir.mkdir(parents=True, exist_ok=True)
        self.today_attendance: dict[str, dict] = {}
        self.current_date: Optional[date] = None
        self._load_today_attendance()

    def _load_today_attendance(self):
        today = date.today()
        if self.current_date == today and self.today_attendance:
            return

        self.current_date = today
        attendance_file = self._get_attendance_file(today)

        if attendance_file.exists():
            try:
                with open(attendance_file, 'r') as f:
                    self.today_attendance = json.load(f)
                logger.info(f"Loaded {len(self.today_attendance)} attendance records for {today}")
            except Exception as e:
                logger.error(f"Failed to load attendance file: {e}")
                self.today_attendance = {}
        else:
            self.today_attendance = {}

    def _get_attendance_file(self, target_date: date) -> Path:
        return self.attendance_dir / f"attendance_{target_date.isoformat()}.json"

    def _save_attendance(self):
        if self.current_date is None:
            return

        attendance_file = self._get_attendance_file(self.current_date)
        try:
            with open(attendance_file, 'w') as f:
                json.dump(self.today_attendance, f, indent=2)
            logger.info(f"Saved attendance for {self.current_date}")
        except Exception as e:
            logger.error(f"Failed to save attendance: {e}")

    def mark_attendance(
        self,
        student_id: str,
        name: str,
        class_name: str,
        confidence: float,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ) -> tuple[bool, str]:
        self._load_today_attendance()

        if student_id in self.today_attendance:
            existing = self.today_attendance[student_id]
            return False, f"Already marked present at {existing['time']}"

        now = datetime.now()
        record = {
            "student_id": student_id,
            "name": name,
            "class": class_name,
            "date": self.current_date.isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "timestamp": now.isoformat(),
            "confidence": confidence,
        }
        if camera_id is not None:
            record["camera_id"] = camera_id
            record["camera_name"] = camera_name or f"Camera {camera_id}"

        self.today_attendance[student_id] = record

        self._save_attendance()
        logger.info(f"Marked attendance for {student_id} ({name})")
        return True, "Attendance marked successfully"

    def get_today_attendance(self) -> list[dict]:
        self._load_today_attendance()
        return list(self.today_attendance.values())

    def get_attendance_by_date(self, target_date: date) -> list[dict]:
        attendance_file = self._get_attendance_file(target_date)
        if not attendance_file.exists():
            return []

        try:
            with open(attendance_file, 'r') as f:
                data = json.load(f)
            return list(data.values())
        except Exception as e:
            logger.error(f"Failed to load attendance for {target_date}: {e}")
            return []

    def get_attendance_range(self, start_date: date, end_date: date) -> list[dict]:
        all_records = []
        current = start_date
        while current <= end_date:
            records = self.get_attendance_by_date(current)
            all_records.extend(records)
            current = date.fromordinal(current.toordinal() + 1)
        return all_records

    def export_to_excel(
        self,
        target_date: Optional[date] = None,
        class_filter: Optional[str] = None,
    ) -> Path:
        export_dir = Path(settings.attendance.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        if target_date is None:
            target_date = date.today()

        records = self.get_attendance_by_date(target_date)

        if class_filter:
            records = [r for r in records if r.get("class") == class_filter]

        if not records:
            raise ValueError("No attendance records found for export")

        df = pd.DataFrame(records)
        columns_order = ["student_id", "name", "class", "date", "time", "confidence"]
        df = df[[col for col in columns_order if col in df.columns]]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_{target_date.isoformat()}_{timestamp}.xlsx"
        if class_filter:
            filename = f"attendance_{target_date.isoformat()}_{class_filter}_{timestamp}.xlsx"

        export_path = export_dir / filename

        df.to_excel(export_path, index=False, engine='openpyxl')
        logger.info(f"Exported attendance to {export_path}")

        return export_path


attendance_service = AttendanceService()
