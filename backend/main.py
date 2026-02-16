import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, enrollment, recognition, attendance, auth, settings as settings_api, cctv, production, analytics
from app.core.config import settings
from app.services.face_engine import FaceEngine
from app.utils.logger import setup_logging

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Face Recognition Attendance System...")
    face_engine = FaceEngine()
    face_engine.initialize()
    app.state.face_engine = face_engine

    # Wire production manager with face engine and attendance service
    from app.services.production_manager import production_manager
    from app.services.attendance_service import attendance_service
    production_manager.initialize_advanced_processing(face_engine)
    production_manager.set_attendance_service(attendance_service)

    logger.info("System ready")
    yield
    logger.info("Shutting down...")
    production_manager.stop_all()
    face_engine.cleanup()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public routes
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(auth.router, prefix="/api", tags=["authentication"])

# Protected routes (requires Bearer token)
from app.api.auth import get_current_user
app.include_router(
    enrollment.router, 
    prefix="/api/enroll", 
    tags=["enrollment"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    recognition.router, 
    prefix="/api/recognition", 
    tags=["recognition"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    attendance.router, 
    prefix="/api/attendance", 
    tags=["attendance"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    settings_api.router, 
    prefix="/api", 
    tags=["settings"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    cctv.router, 
    prefix="/api", 
    tags=["cctv"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    production.router,
    prefix="/api",
    tags=["production"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    analytics.router,
    prefix="/api/analytics",
    tags=["analytics"],
    dependencies=[Depends(get_current_user)]
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
