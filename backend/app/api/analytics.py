import logging
from collections import Counter, defaultdict
from datetime import date, timedelta
from statistics import median

from fastapi import APIRouter, Request, Query, HTTPException

from app.services.attendance_service import attendance_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_records(days: int) -> list[dict]:
    end = date.today()
    start = end - timedelta(days=days)
    return attendance_service.get_attendance_range(start, end)


@router.get("/overview")
async def get_overview(request: Request, days: int = Query(30, ge=1, le=365)):
    try:
        records = _get_records(days)

        student_ids = set()
        classes = set()
        dates_seen = set()
        confidences = []

        for r in records:
            student_ids.add(r.get("student_id"))
            classes.add(r.get("class", ""))
            dates_seen.add(r.get("date", ""))
            c = r.get("confidence")
            if c is not None:
                confidences.append(float(c))

        total_days = len(dates_seen) or 1
        avg_daily = round(len(records) / total_days, 1)
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        # Enrolled students info from face engine
        face_engine = request.app.state.face_engine
        enrolled_classes: dict[str, int] = defaultdict(int)
        for sid, meta in face_engine.gallery_metadata.items():
            cls = meta.get("class_name", "unknown")
            enrolled_classes[cls] += 1

        return {
            "total_records": len(records),
            "unique_students": len(student_ids),
            "total_days": len(dates_seen),
            "avg_daily_attendance": avg_daily,
            "avg_confidence": avg_confidence,
            "classes": sorted(classes - {""}),
            "enrolled_count": len(face_engine.gallery_metadata),
            "enrolled_classes": dict(enrolled_classes),
            "days": days,
        }
    except Exception as e:
        logger.error(f"Overview analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_trends(days: int = Query(30, ge=1, le=365)):
    try:
        records = _get_records(days)

        counts_by_date: dict[str, int] = defaultdict(int)
        for r in records:
            counts_by_date[r.get("date", "")] += 1

        # Build complete date range with zeros for missing days
        end = date.today()
        start = end - timedelta(days=days)
        dates = []
        counts = []
        current = start
        while current <= end:
            d = current.isoformat()
            dates.append(d)
            counts.append(counts_by_date.get(d, 0))
            current += timedelta(days=1)

        total = sum(counts)
        avg = round(total / len(counts), 1) if counts else 0.0

        # Trend: compare first half vs second half
        mid = len(counts) // 2
        first_half = sum(counts[:mid]) / max(mid, 1)
        second_half = sum(counts[mid:]) / max(len(counts) - mid, 1)
        if second_half > first_half * 1.1:
            trend = "increasing"
        elif second_half < first_half * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "dates": dates,
            "counts": counts,
            "average": avg,
            "trend": trend,
        }
    except Exception as e:
        logger.error(f"Trends analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/class-breakdown")
async def get_class_breakdown(days: int = Query(30, ge=1, le=365)):
    try:
        records = _get_records(days)

        class_data: dict[str, dict] = defaultdict(lambda: {
            "students": set(),
            "dates": set(),
            "total": 0,
        })

        for r in records:
            cls = r.get("class", "unknown")
            class_data[cls]["students"].add(r.get("student_id"))
            class_data[cls]["dates"].add(r.get("date", ""))
            class_data[cls]["total"] += 1

        result = []
        for name, data in sorted(class_data.items()):
            days_active = len(data["dates"])
            result.append({
                "name": name,
                "total_records": data["total"],
                "unique_students": len(data["students"]),
                "avg_daily": round(data["total"] / max(days_active, 1), 1),
                "days_active": days_active,
            })

        return {"classes": result}
    except Exception as e:
        logger.error(f"Class breakdown error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students")
async def get_students(request: Request, days: int = Query(30, ge=1, le=365)):
    try:
        records = _get_records(days)

        # Count total unique days with any attendance
        all_dates = set()
        for r in records:
            all_dates.add(r.get("date", ""))
        total_days = len(all_dates) or 1

        # Per-student stats
        student_data: dict[str, dict] = defaultdict(lambda: {
            "name": "",
            "class": "",
            "dates": set(),
            "confidences": [],
            "last_seen": "",
        })

        for r in records:
            sid = r.get("student_id", "")
            sd = student_data[sid]
            sd["name"] = r.get("name", "")
            sd["class"] = r.get("class", "")
            sd["dates"].add(r.get("date", ""))
            c = r.get("confidence")
            if c is not None:
                sd["confidences"].append(float(c))
            d = r.get("date", "")
            if d > sd["last_seen"]:
                sd["last_seen"] = d

        # Also include enrolled students who have zero attendance
        face_engine = request.app.state.face_engine
        for sid, meta in face_engine.gallery_metadata.items():
            if sid not in student_data:
                student_data[sid] = {
                    "name": meta.get("name", ""),
                    "class": meta.get("class_name", ""),
                    "dates": set(),
                    "confidences": [],
                    "last_seen": "Never",
                }

        students = []
        for sid, sd in student_data.items():
            days_present = len(sd["dates"])
            confs = sd["confidences"]
            students.append({
                "student_id": sid,
                "name": sd["name"],
                "class": sd["class"],
                "days_present": days_present,
                "total_days": total_days,
                "attendance_rate": round((days_present / total_days) * 100, 1),
                "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
                "last_seen": sd["last_seen"],
            })

        students.sort(key=lambda s: s["attendance_rate"], reverse=True)

        return {
            "students": students,
            "total_enrolled": len(face_engine.gallery_metadata),
        }
    except Exception as e:
        logger.error(f"Students analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hourly")
async def get_hourly(days: int = Query(30, ge=1, le=365)):
    try:
        records = _get_records(days)

        hour_counts: dict[int, int] = defaultdict(int)
        for r in records:
            time_str = r.get("time", "")
            if time_str:
                try:
                    hour = int(time_str.split(":")[0])
                    hour_counts[hour] += 1
                except (ValueError, IndexError):
                    pass

        # Fill all 24 hours
        distribution = {str(h): hour_counts.get(h, 0) for h in range(24)}

        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0
        peak_count = hour_counts.get(peak_hour, 0)

        return {
            "distribution": distribution,
            "peak_hour": peak_hour,
            "peak_count": peak_count,
        }
    except Exception as e:
        logger.error(f"Hourly analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confidence")
async def get_confidence(days: int = Query(30, ge=1, le=365)):
    try:
        records = _get_records(days)

        confidences = []
        for r in records:
            c = r.get("confidence")
            if c is not None:
                confidences.append(float(c))

        if not confidences:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "distribution": [
                    {"range": "0-30%", "count": 0},
                    {"range": "30-50%", "count": 0},
                    {"range": "50-70%", "count": 0},
                    {"range": "70-90%", "count": 0},
                    {"range": "90-100%", "count": 0},
                ],
            }

        # Histogram buckets
        buckets = [0, 0, 0, 0, 0]
        for c in confidences:
            if c < 0.3:
                buckets[0] += 1
            elif c < 0.5:
                buckets[1] += 1
            elif c < 0.7:
                buckets[2] += 1
            elif c < 0.9:
                buckets[3] += 1
            else:
                buckets[4] += 1

        return {
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
            "mean": round(sum(confidences) / len(confidences), 3),
            "median": round(median(confidences), 3),
            "distribution": [
                {"range": "0-30%", "count": buckets[0]},
                {"range": "30-50%", "count": buckets[1]},
                {"range": "50-70%", "count": buckets[2]},
                {"range": "70-90%", "count": buckets[3]},
                {"range": "90-100%", "count": buckets[4]},
            ],
        }
    except Exception as e:
        logger.error(f"Confidence analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
