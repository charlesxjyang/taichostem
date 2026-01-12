"""FastAPI backend for 4D-STEM visualization platform."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    health_router,
    dataset_router,
    analysis_router,
    calibration_router,
    preprocessing_router,
)

app = FastAPI(title="4D-STEM Viewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(dataset_router)
app.include_router(analysis_router)
app.include_router(calibration_router)
app.include_router(preprocessing_router)
