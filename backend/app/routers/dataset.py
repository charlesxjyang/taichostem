"""Dataset loading and diffraction pattern endpoints."""

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import py4DSTEM
from fastapi import APIRouter, HTTPException

from app.schemas.dataset import (
    DatasetProbeRequest,
    DatasetLoadRequest,
    DatasetLoadResponse,
    SingleProbeResponse,
    HDF5TreeProbeResponse,
)
from app.schemas.diffraction import (
    MeanDiffractionResponse,
    MaxDiffractionResponse,
    VirtualImageResponse,
    DiffractionPatternResponse,
    RegionDiffractionRequest,
    RegionDiffractionResponse,
)
from app.services.state import (
    get_current_dataset,
    set_current_dataset,
    get_cached_mean_diffraction,
    get_cached_max_diffraction,
    get_current_virtual_image,
    set_current_virtual_image,
    compute_diffraction_stats,
)
from app.services.image_processing import (
    process_image_for_display,
    create_region_mask,
    image_to_base64_png,
)
from app.services.hdf5 import walk_hdf5_tree, build_hdf5_tree

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.post("/probe")
async def probe_dataset(
    request: DatasetProbeRequest,
) -> SingleProbeResponse | HDF5TreeProbeResponse:
    """
    Probe a dataset file to determine its structure.

    For single-datacube formats (.dm4, .mrc), returns basic metadata.
    For HDF5 formats (.h5, .hdf5, .emd), walks the tree and returns
    information about all array datasets.

    Parameters
    ----------
    request : DatasetProbeRequest
        Request containing the path to the dataset file.

    Returns
    -------
    SingleProbeResponse | HDF5TreeProbeResponse
        Either single datacube metadata or HDF5 tree structure.

    Raises
    ------
    HTTPException
        404 if file not found, 400 if file format unsupported.
    """
    file_path = Path(request.path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {request.path}")

    suffix = file_path.suffix.lower()
    hdf5_extensions = {".h5", ".hdf5", ".emd"}
    single_extensions = {".dm4", ".mrc"}
    supported_extensions = hdf5_extensions | single_extensions

    if suffix not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}",
        )

    # Handle HDF5 files - walk the tree
    if suffix in hdf5_extensions:
        try:
            with h5py.File(str(file_path), "r") as h5file:
                datasets = walk_hdf5_tree(h5file)
                tree = build_hdf5_tree(h5file)
            return HDF5TreeProbeResponse(
                type="hdf5_tree",
                filename=file_path.name,
                root=tree,
                datasets=datasets,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to probe HDF5 file: {str(e)}",
            )

    # Handle single-datacube formats (.dm4, .mrc)
    try:
        data = py4DSTEM.read(str(file_path))

        if isinstance(data, py4DSTEM.DataCube):
            datacube = data
        elif hasattr(data, "data"):
            datacube = data
        else:
            datacube = data

        if hasattr(datacube, "data"):
            shape = list(datacube.data.shape)
            dtype = str(datacube.data.dtype)
        else:
            shape = list(datacube.shape)
            dtype = str(datacube.dtype)

        return SingleProbeResponse(type="single", shape=shape, dtype=dtype)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to probe dataset: {str(e)}",
        )


@router.post("/load", response_model=DatasetLoadResponse)
async def load_dataset(request: DatasetLoadRequest) -> DatasetLoadResponse:
    """
    Load a 4D-STEM dataset and return its metadata.

    Parameters
    ----------
    request : DatasetLoadRequest
        Request containing the path to the dataset file and optionally
        the internal dataset path for HDF5 files.

    Returns
    -------
    DatasetLoadResponse
        Dataset metadata including shape [Rx, Ry, Qx, Qy] and dtype.

    Raises
    ------
    HTTPException
        404 if file not found, 400 if file format unsupported or
        multiple datasets found without dataset_path specified,
        500 for other loading errors.
    """
    file_path = Path(request.path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.path}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {request.path}")

    suffix = file_path.suffix.lower()
    hdf5_extensions = {".h5", ".hdf5", ".emd"}
    single_extensions = {".dm4", ".mrc"}
    supported_extensions = hdf5_extensions | single_extensions

    if suffix not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}",
        )

    # Handle HDF5 files with dataset_path parameter
    if suffix in hdf5_extensions:
        try:
            with h5py.File(str(file_path), "r") as h5file:
                if request.dataset_path:
                    # Load specific dataset
                    if request.dataset_path not in h5file:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Dataset not found: {request.dataset_path}",
                        )
                    dataset = h5file[request.dataset_path]
                    if not isinstance(dataset, h5py.Dataset):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Path is not a dataset: {request.dataset_path}",
                        )
                    # Cache the dataset in memory
                    data = dataset[:]
                    set_current_dataset(data, f"{request.path}:{request.dataset_path}")
                    compute_diffraction_stats()
                    shape = list(dataset.shape)
                    dtype = str(dataset.dtype)
                    return DatasetLoadResponse(
                        shape=shape,
                        dtype=dtype,
                        dataset_path=request.dataset_path,
                    )
                else:
                    # No dataset_path provided - check for multiple 4D datasets
                    datasets = walk_hdf5_tree(h5file)
                    datasets_4d = [d for d in datasets if d.is_4d]

                    if len(datasets_4d) > 1:
                        raise HTTPException(
                            status_code=400,
                            detail="Multiple datasets found, please specify dataset_path",
                        )
                    elif len(datasets_4d) == 1:
                        # Auto-select the only 4D dataset
                        ds_info = datasets_4d[0]
                        dataset = h5file[ds_info.path]
                        data = dataset[:]
                        set_current_dataset(data, f"{request.path}:{ds_info.path}")
                        compute_diffraction_stats()
                        return DatasetLoadResponse(
                            shape=ds_info.shape,
                            dtype=ds_info.dtype,
                            dataset_path=ds_info.path,
                        )
                    elif len(datasets) == 1:
                        # Only one dataset exists (not 4D), use it
                        ds_info = datasets[0]
                        dataset = h5file[ds_info.path]
                        data = dataset[:]
                        set_current_dataset(data, f"{request.path}:{ds_info.path}")
                        compute_diffraction_stats()
                        return DatasetLoadResponse(
                            shape=ds_info.shape,
                            dtype=ds_info.dtype,
                            dataset_path=ds_info.path,
                        )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="No suitable datasets found in HDF5 file",
                        )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load HDF5 dataset: {str(e)}",
            )

    # Handle single-datacube formats (.dm4, .mrc) - ignore dataset_path
    try:
        data = py4DSTEM.read(str(file_path))

        # py4DSTEM.read() returns different types depending on the file
        datacube = None

        if isinstance(data, py4DSTEM.io.datastructure.Root):
            # Search for a DataCube in the Root's children
            for item in data._branch.values():
                if isinstance(item, py4DSTEM.DataCube):
                    datacube = item
                    break
            if datacube is None:
                raise ValueError("No DataCube found in file")
        elif isinstance(data, py4DSTEM.DataCube):
            datacube = data
        else:
            datacube = data

        if hasattr(datacube, "data"):
            dataset_data = datacube.data[:]
            shape = list(datacube.data.shape)
            dtype = str(datacube.data.dtype)
        else:
            dataset_data = datacube[:]
            shape = list(datacube.shape)
            dtype = str(datacube.dtype)

        set_current_dataset(dataset_data, request.path)
        compute_diffraction_stats()

        return DatasetLoadResponse(shape=shape, dtype=dtype, dataset_path=None)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset: {str(e)}",
        )


@router.get("/diffraction/mean", response_model=MeanDiffractionResponse)
async def get_mean_diffraction(
    log_scale: bool = False,
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> MeanDiffractionResponse:
    """
    Get the mean diffraction pattern across all scan positions.

    Computes the mean of the 4D-STEM dataset over the real-space dimensions
    (Rx, Ry), returning the average diffraction pattern as a base64-encoded
    grayscale PNG.

    Parameters
    ----------
    log_scale : bool
        If True, apply log10 transform before display.
    contrast_min : float
        Minimum percentile for contrast stretch (0-100).
    contrast_max : float
        Maximum percentile for contrast stretch (0-100).

    Returns
    -------
    MeanDiffractionResponse
        Base64-encoded PNG image and dimensions.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded.
    """
    cached_mean = get_cached_mean_diffraction()
    if cached_mean is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    normalized = process_image_for_display(
        cached_mean, log_scale, contrast_min, contrast_max
    )
    image_base64 = image_to_base64_png(normalized)

    return MeanDiffractionResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
    )


@router.get("/diffraction/max", response_model=MaxDiffractionResponse)
async def get_max_diffraction(
    log_scale: bool = False,
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> MaxDiffractionResponse:
    """
    Get the max diffraction pattern across all scan positions.

    Computes the maximum intensity at each diffraction pixel across all
    real-space positions, returning a pattern that shows all possible
    diffraction features. Useful for polycrystalline samples.

    Parameters
    ----------
    log_scale : bool
        If True, apply log10 transform before display.
    contrast_min : float
        Minimum percentile for contrast stretch (0-100).
    contrast_max : float
        Maximum percentile for contrast stretch (0-100).

    Returns
    -------
    MaxDiffractionResponse
        Base64-encoded PNG image and dimensions.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded.
    """
    cached_max = get_cached_max_diffraction()
    if cached_max is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    normalized = process_image_for_display(
        cached_max, log_scale, contrast_min, contrast_max
    )
    image_base64 = image_to_base64_png(normalized)

    return MaxDiffractionResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
    )


@router.get("/virtual-image", response_model=VirtualImageResponse)
async def get_virtual_image(
    type: Literal["bf", "abf", "adf"] = "bf",
    inner: int = 0,
    outer: int = 20,
    log_scale: bool = False,
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> VirtualImageResponse:
    """
    Compute a virtual image by integrating within a circular/annular detector.

    Uses py4DSTEM's get_virtual_image method for efficient computation.

    Supports three detector modes:
    - BF (bright-field): Center disk, inner=0, uses only outer radius
    - ABF (annular bright-field): Annular region just outside BF disk
    - ADF (annular dark-field): Outer annular ring

    Parameters
    ----------
    type : Literal["bf", "abf", "adf"]
        Detector type.
    inner : int
        Inner radius of the detector in pixels (ignored for BF mode).
    outer : int
        Outer radius of the detector in pixels.
    log_scale : bool
        If True, apply log10 transform before display.
    contrast_min : float
        Minimum percentile for contrast stretch (0-100).
    contrast_max : float
        Maximum percentile for contrast stretch (0-100).

    Returns
    -------
    VirtualImageResponse
        Base64-encoded grayscale PNG image and dimensions.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded or invalid parameters.
    """
    # For BF mode, force inner to 0
    if type == "bf":
        inner = 0

    current_dataset = get_current_dataset()
    if current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    if inner < 0 or outer <= 0 or inner >= outer:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid radii: inner={inner}, outer={outer}. "
            "Require: 0 <= inner < outer",
        )

    # Get diffraction pattern dimensions
    if current_dataset.ndim == 4:
        _, _, qx, qy = current_dataset.shape
    elif current_dataset.ndim == 3:
        _, qx, qy = current_dataset.shape
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be 3D or 4D, got {current_dataset.ndim}D",
        )

    try:
        # Create py4DSTEM DataCube from current dataset
        datacube = py4DSTEM.DataCube(data=current_dataset)

        # Calculate center of diffraction pattern
        center_x, center_y = qx / 2, qy / 2

        if type == "bf":
            # Bright-field: circular detector
            virtual_image = datacube.get_virtual_image(
                mode="circle",
                geometry=((center_x, center_y), outer),
            )
        else:
            # ABF or ADF: annular detector
            virtual_image = datacube.get_virtual_image(
                mode="annulus",
                geometry=((center_x, center_y), (inner, outer)),
            )

        # Extract the data array from the VirtualImage object
        if hasattr(virtual_image, "data"):
            virtual_image = virtual_image.data

        # Store raw virtual image for atom detection
        set_current_virtual_image(virtual_image.copy())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute virtual image: {str(e)}",
        )

    normalized = process_image_for_display(
        virtual_image, log_scale, contrast_min, contrast_max
    )
    image_base64 = image_to_base64_png(normalized)

    return VirtualImageResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
    )


@router.get("/diffraction", response_model=DiffractionPatternResponse)
async def get_diffraction_pattern(
    x: int,
    y: int,
    log_scale: bool = False,
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> DiffractionPatternResponse:
    """
    Get the diffraction pattern at a specific scan position.

    Parameters
    ----------
    x : int
        X coordinate in the scan (real space) grid.
    y : int
        Y coordinate in the scan (real space) grid.
    log_scale : bool
        If True, apply log10 transform before display.
    contrast_min : float
        Minimum percentile for contrast stretch (0-100).
    contrast_max : float
        Maximum percentile for contrast stretch (0-100).

    Returns
    -------
    DiffractionPatternResponse
        Base64-encoded grayscale PNG of the diffraction pattern and dimensions.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded or coordinates are out of bounds.
    """
    current_dataset = get_current_dataset()
    if current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Validate coordinates
    if current_dataset.ndim == 4:
        rx, ry, qx, qy = current_dataset.shape
    elif current_dataset.ndim == 3:
        rx, qx, qy = current_dataset.shape
        ry = 1
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be 3D or 4D, got {current_dataset.ndim}D",
        )

    if x < 0 or x >= rx or y < 0 or y >= ry:
        raise HTTPException(
            status_code=400,
            detail=f"Coordinates ({x}, {y}) out of bounds. "
            f"Valid range: x=[0, {rx-1}], y=[0, {ry-1}]",
        )

    # Extract diffraction pattern at (x, y)
    if current_dataset.ndim == 4:
        pattern = current_dataset[x, y, :, :]
    else:
        pattern = current_dataset[x, :, :]

    normalized = process_image_for_display(
        pattern, log_scale, contrast_min, contrast_max
    )
    image_base64 = image_to_base64_png(normalized)

    return DiffractionPatternResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
        x=x,
        y=y,
    )


@router.post("/diffraction/region", response_model=RegionDiffractionResponse)
async def get_region_diffraction(
    request: RegionDiffractionRequest,
) -> RegionDiffractionResponse:
    """
    Get mean or max diffraction pattern for a selected region.

    Computes the mean or max of diffraction patterns only for pixels
    within the specified region in real space.

    Parameters
    ----------
    request : RegionDiffractionRequest
        mode : "mean" or "max"
        region_type : "rectangle", "ellipse", or "polygon"
        points : Coordinates defining the region
        log_scale : Apply log10 transform before display
        contrast_min : Minimum percentile for contrast stretch
        contrast_max : Maximum percentile for contrast stretch

    Returns
    -------
    RegionDiffractionResponse
        Base64-encoded PNG image and metadata.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded or region is empty.
    """
    current_dataset = get_current_dataset()
    if current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Get real space dimensions
    if current_dataset.ndim == 4:
        rx, ry, qx, qy = current_dataset.shape
    elif current_dataset.ndim == 3:
        rx, qx, qy = current_dataset.shape
        ry = 1
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be 3D or 4D, got {current_dataset.ndim}D",
        )

    # Create mask for the region
    mask = create_region_mask(request.region_type, request.points, (rx, ry))
    pixels_in_region = int(np.sum(mask))

    if pixels_in_region == 0:
        raise HTTPException(
            status_code=400,
            detail="Selected region contains no pixels",
        )

    # Compute diffraction pattern using py4DSTEM with fallback to NumPy
    try:
        datacube = py4DSTEM.DataCube(data=current_dataset)

        virtual_diffraction = datacube.get_virtual_diffraction(
            method=request.mode,
            mask=mask,
            name=f"{request.region_type}_{request.mode}_diffraction",
        )

        if hasattr(virtual_diffraction, "data"):
            result = virtual_diffraction.data
        else:
            result = virtual_diffraction

    except Exception:
        # Fallback to NumPy if py4DSTEM method fails
        if current_dataset.ndim == 4:
            indices = np.argwhere(mask)
            patterns = np.array([current_dataset[i, j, :, :] for i, j in indices])
        else:
            indices = np.argwhere(mask[:, 0])
            patterns = np.array([current_dataset[i, :, :] for i in indices.flatten()])

        if request.mode == "mean":
            result = np.mean(patterns, axis=0)
        else:
            result = np.max(patterns, axis=0)

    normalized = process_image_for_display(
        result, request.log_scale, request.contrast_min, request.contrast_max
    )
    image_base64 = image_to_base64_png(normalized)

    return RegionDiffractionResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
        mode=request.mode,
        region_type=request.region_type,
        pixels_in_region=pixels_in_region,
    )
