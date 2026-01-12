"""API routers for the 4D-STEM API."""

from app.routers.health import router as health_router
from app.routers.dataset import router as dataset_router
from app.routers.analysis import router as analysis_router
from app.routers.calibration import router as calibration_router
from app.routers.preprocessing import router as preprocessing_router

__all__ = [
    "health_router",
    "dataset_router",
    "analysis_router",
    "calibration_router",
    "preprocessing_router",
]
