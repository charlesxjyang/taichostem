"""Calibration endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.calibration import CalibrationResponse, CalibrationUpdateRequest
from app.services.state import (
    get_current_dataset,
    get_calibration,
    update_calibration,
)

router = APIRouter(prefix="/dataset", tags=["calibration"])


@router.get("/calibration", response_model=CalibrationResponse)
async def get_calibration_endpoint() -> CalibrationResponse:
    """
    Get the current calibration values for the loaded dataset.

    Returns the pixel sizes and units for both real space (R) and
    reciprocal space (Q).

    Returns
    -------
    CalibrationResponse
        Current calibration values.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded.
    """
    if get_current_dataset() is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    calibration = get_calibration()
    return CalibrationResponse(
        q_pixel_size=float(calibration["q_pixel_size"]),
        q_pixel_units=str(calibration["q_pixel_units"]),
        r_pixel_size=float(calibration["r_pixel_size"]),
        r_pixel_units=str(calibration["r_pixel_units"]),
    )


@router.post("/calibration", response_model=CalibrationResponse)
async def set_calibration_endpoint(request: CalibrationUpdateRequest) -> CalibrationResponse:
    """
    Update the calibration values for the loaded dataset.

    Only the provided fields will be updated; others remain unchanged.

    Parameters
    ----------
    request : CalibrationUpdateRequest
        Fields to update. All fields are optional.

    Returns
    -------
    CalibrationResponse
        Updated calibration values.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded or invalid values provided.
    """
    if get_current_dataset() is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Validate pixel sizes
    if request.q_pixel_size is not None and request.q_pixel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="q_pixel_size must be positive",
        )

    if request.r_pixel_size is not None and request.r_pixel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="r_pixel_size must be positive",
        )

    calibration = update_calibration(
        q_pixel_size=request.q_pixel_size,
        q_pixel_units=request.q_pixel_units,
        r_pixel_size=request.r_pixel_size,
        r_pixel_units=request.r_pixel_units,
    )

    return CalibrationResponse(
        q_pixel_size=float(calibration["q_pixel_size"]),
        q_pixel_units=str(calibration["q_pixel_units"]),
        r_pixel_size=float(calibration["r_pixel_size"]),
        r_pixel_units=str(calibration["r_pixel_units"]),
    )
