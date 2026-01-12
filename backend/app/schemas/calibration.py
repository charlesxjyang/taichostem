"""Calibration-related Pydantic schemas."""

from pydantic import BaseModel


class CalibrationResponse(BaseModel):
    """Response model for calibration values."""

    q_pixel_size: float
    q_pixel_units: str
    r_pixel_size: float
    r_pixel_units: str


class CalibrationUpdateRequest(BaseModel):
    """Request model for updating calibration values."""

    q_pixel_size: float | None = None
    q_pixel_units: str | None = None
    r_pixel_size: float | None = None
    r_pixel_units: str | None = None
