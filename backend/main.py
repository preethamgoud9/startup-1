import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, enrollment, recognition, attendance
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
    logger.info("System ready")
    yield
    logger.info("Shutting down...")
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

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(enrollment.router, prefix="/api/enroll", tags=["enrollment"])
app.include_router(recognition.router, prefix="/api/recognition", tags=["recognition"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["attendance"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
