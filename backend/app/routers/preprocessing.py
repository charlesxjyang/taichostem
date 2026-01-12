"""Preprocessing endpoints (hot pixel filtering, etc.)."""

import numpy as np
import py4DSTEM
from fastapi import APIRouter, HTTPException

from app.schemas.analysis import FilterHotPixelsRequest, FilterHotPixelsResponse
from app.services.state import (
    get_current_dataset,
    set_current_dataset,
    set_current_virtual_image,
    compute_diffraction_stats,
)

router = APIRouter(prefix="/dataset/preprocess", tags=["preprocessing"])


@router.post("/filter-hot-pixels", response_model=FilterHotPixelsResponse)
async def filter_hot_pixels(request: FilterHotPixelsRequest) -> FilterHotPixelsResponse:
    """
    Filter hot pixels from the current 4D-STEM dataset.

    Uses py4DSTEM's filter_hot_pixels method to identify and remove
    anomalously bright pixels that can interfere with analysis.

    Parameters
    ----------
    request : FilterHotPixelsRequest
        thresh : float
            Threshold for hot pixel detection. Pixels with intensity
            more than thresh standard deviations above the mean are
            replaced. Default is 8.0.

    Returns
    -------
    FilterHotPixelsResponse
        success : bool
            Whether the operation completed successfully.
        pixels_filtered : int
            Number of hot pixels that were filtered.
        message : str
            Status message describing the result.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded.
        500 if filtering fails.
    """
    current_dataset = get_current_dataset()
    if current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    try:
        datacube = py4DSTEM.DataCube(data=current_dataset)

        # Count pixels before filtering (estimate hot pixels by threshold)
        mean_val = np.mean(current_dataset)
        std_val = np.std(current_dataset)
        hot_pixel_mask = current_dataset > (mean_val + request.thresh * std_val)
        pixels_before = int(np.sum(hot_pixel_mask))

        # Apply hot pixel filter
        datacube.filter_hot_pixels(thresh=request.thresh)

        # Update the in-memory dataset with the filtered data
        set_current_dataset(datacube.data)
        compute_diffraction_stats()
        set_current_virtual_image(None)

        return FilterHotPixelsResponse(
            success=True,
            pixels_filtered=pixels_before,
            message=f"Successfully filtered hot pixels with threshold {request.thresh}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to filter hot pixels: {str(e)}",
        )
