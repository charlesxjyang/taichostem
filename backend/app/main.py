"""FastAPI backend for 4D-STEM visualization platform."""

import base64
import io
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import py4DSTEM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="4D-STEM Viewer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- In-memory dataset cache ---
# Stores the currently loaded 4D-STEM data array for fast access
_current_dataset: np.ndarray | None = None
_current_dataset_path: str | None = None


# --- Probe Endpoint Schemas ---


class DatasetProbeRequest(BaseModel):
    """Request model for probing a dataset file."""

    path: str


class HDF5DatasetInfo(BaseModel):
    """Information about a single dataset within an HDF5 file."""

    path: str
    shape: list[int]
    dtype: str
    is_4d: bool


class SingleProbeResponse(BaseModel):
    """Response for single-datacube files (.dm4, .mrc)."""

    type: Literal["single"]
    shape: list[int]
    dtype: str


class HDF5TreeProbeResponse(BaseModel):
    """Response for HDF5 files with a tree structure."""

    type: Literal["hdf5_tree"]
    datasets: list[HDF5DatasetInfo]


# --- Load Endpoint Schemas ---


class DatasetLoadRequest(BaseModel):
    """Request model for loading a dataset."""

    path: str
    dataset_path: str | None = None


class DatasetLoadResponse(BaseModel):
    """Response model for loaded dataset metadata."""

    shape: list[int]
    dtype: str
    dataset_path: str | None = None


class MeanDiffractionResponse(BaseModel):
    """Response model for mean diffraction pattern."""

    image_base64: str
    width: int
    height: int


class VirtualImageResponse(BaseModel):
    """Response model for virtual image."""

    image_base64: str
    width: int
    height: int


class DiffractionPatternResponse(BaseModel):
    """Response model for diffraction pattern at a specific position."""

    image_base64: str
    width: int
    height: int
    x: int
    y: int


def _process_image_for_display(
    data: np.ndarray,
    log_scale: bool = False,
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> np.ndarray:
    """
    Process image data for display with optional log scale and contrast adjustment.

    Parameters
    ----------
    data : np.ndarray
        Input image data (2D array).
    log_scale : bool
        If True, apply log10 transform (with offset to handle zeros).
    contrast_min : float
        Minimum percentile for contrast stretch (0-100).
    contrast_max : float
        Maximum percentile for contrast stretch (0-100).

    Returns
    -------
    np.ndarray
        Processed image as uint8 (0-255).
    """
    # Convert to float for processing
    result = data.astype(np.float64)

    # Apply log scale if requested
    if log_scale:
        # Add small offset to handle zeros, then take log10
        min_positive = np.min(result[result > 0]) if np.any(result > 0) else 1.0
        result = np.log10(result + min_positive * 0.1)

    # Apply percentile-based contrast stretching
    p_min = np.percentile(result, contrast_min)
    p_max = np.percentile(result, contrast_max)

    if p_max > p_min:
        # Clip to percentile range and normalize to 0-255
        result = np.clip(result, p_min, p_max)
        normalized = ((result - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(result, dtype=np.uint8)

    return normalized


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _walk_hdf5_tree(h5file: h5py.File) -> list[HDF5DatasetInfo]:
    """
    Recursively walk an HDF5 file and collect dataset information.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    list[HDF5DatasetInfo]
        List of dataset information for all array datasets (2D or higher).
    """
    datasets: list[HDF5DatasetInfo] = []

    def visitor(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            # Only include datasets with 2 or more dimensions
            if len(shape) >= 2:
                datasets.append(
                    HDF5DatasetInfo(
                        path=f"/{name}",
                        shape=list(shape),
                        dtype=str(obj.dtype),
                        is_4d=len(shape) == 4,
                    )
                )

    h5file.visititems(visitor)
    return datasets


@app.post("/dataset/probe")
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
                datasets = _walk_hdf5_tree(h5file)
            return HDF5TreeProbeResponse(type="hdf5_tree", datasets=datasets)
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


@app.post("/dataset/load", response_model=DatasetLoadResponse)
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

    global _current_dataset, _current_dataset_path

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
                    _current_dataset = dataset[:]
                    _current_dataset_path = f"{request.path}:{request.dataset_path}"
                    shape = list(dataset.shape)
                    dtype = str(dataset.dtype)
                    return DatasetLoadResponse(
                        shape=shape,
                        dtype=dtype,
                        dataset_path=request.dataset_path,
                    )
                else:
                    # No dataset_path provided - check for multiple 4D datasets
                    datasets = _walk_hdf5_tree(h5file)
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
                        _current_dataset = dataset[:]
                        _current_dataset_path = f"{request.path}:{ds_info.path}"
                        return DatasetLoadResponse(
                            shape=ds_info.shape,
                            dtype=ds_info.dtype,
                            dataset_path=ds_info.path,
                        )
                    elif len(datasets) == 1:
                        # Only one dataset exists (not 4D), use it
                        ds_info = datasets[0]
                        dataset = h5file[ds_info.path]
                        _current_dataset = dataset[:]
                        _current_dataset_path = f"{request.path}:{ds_info.path}"
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

        # py4DSTEM.read() returns different types depending on the file:
        # - Root object for HDF5 files (container with multiple objects)
        # - DataCube directly for DM4 files
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
            # Fallback: try to use the object directly if it has shape
            datacube = data

        if hasattr(datacube, "data"):
            _current_dataset = datacube.data[:]
            shape = list(datacube.data.shape)
            dtype = str(datacube.data.dtype)
        else:
            _current_dataset = datacube[:]
            shape = list(datacube.shape)
            dtype = str(datacube.dtype)

        _current_dataset_path = request.path

        return DatasetLoadResponse(shape=shape, dtype=dtype, dataset_path=None)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset: {str(e)}",
        )


@app.get("/dataset/diffraction/mean", response_model=MeanDiffractionResponse)
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
    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Compute mean over real-space dimensions (first two axes for 4D data)
    # Shape is typically [Rx, Ry, Qx, Qy]
    if _current_dataset.ndim == 4:
        mean_pattern = np.mean(_current_dataset, axis=(0, 1))
    elif _current_dataset.ndim == 3:
        mean_pattern = np.mean(_current_dataset, axis=0)
    else:
        mean_pattern = _current_dataset

    # Process for display with log scale and contrast
    normalized = _process_image_for_display(
        mean_pattern, log_scale, contrast_min, contrast_max
    )

    # Convert to PNG
    image = Image.fromarray(normalized, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return MeanDiffractionResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
    )


@app.get("/dataset/virtual-image", response_model=VirtualImageResponse)
async def get_virtual_image(
    inner: int = 0,
    outer: int = 20,
    log_scale: bool = False,
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> VirtualImageResponse:
    """
    Compute a virtual bright-field image by integrating within an annular detector.

    Integrates the diffraction pattern intensity within a circular/annular region
    defined by inner and outer radii (in pixels from the center).

    Parameters
    ----------
    inner : int
        Inner radius of the annular detector in pixels (default 0 for bright-field).
    outer : int
        Outer radius of the annular detector in pixels (default 20).
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
    if _current_dataset is None:
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
    if _current_dataset.ndim == 4:
        rx, ry, qx, qy = _current_dataset.shape
    elif _current_dataset.ndim == 3:
        rx, qx, qy = _current_dataset.shape
        ry = 1
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be 3D or 4D, got {_current_dataset.ndim}D",
        )

    # Create circular mask for the annular detector
    center_x, center_y = qx // 2, qy // 2
    y_coords, x_coords = np.ogrid[:qx, :qy]
    distance_from_center = np.sqrt(
        (x_coords - center_y) ** 2 + (y_coords - center_x) ** 2
    )
    mask = (distance_from_center >= inner) & (distance_from_center < outer)

    # Compute virtual image by integrating over the masked region
    if _current_dataset.ndim == 4:
        # Sum over masked diffraction pixels for each scan position
        virtual_image = np.sum(
            _current_dataset[:, :, mask], axis=-1
        )
    else:
        # 3D case: reshape to 2D for output
        virtual_image = np.sum(
            _current_dataset[:, mask], axis=-1
        ).reshape(rx, 1)

    # Process for display with log scale and contrast
    normalized = _process_image_for_display(
        virtual_image, log_scale, contrast_min, contrast_max
    )

    # Convert to PNG
    image = Image.fromarray(normalized, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return VirtualImageResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
    )


@app.get("/dataset/diffraction", response_model=DiffractionPatternResponse)
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
    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Validate coordinates
    if _current_dataset.ndim == 4:
        rx, ry, qx, qy = _current_dataset.shape
    elif _current_dataset.ndim == 3:
        rx, qx, qy = _current_dataset.shape
        ry = 1
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be 3D or 4D, got {_current_dataset.ndim}D",
        )

    if x < 0 or x >= rx or y < 0 or y >= ry:
        raise HTTPException(
            status_code=400,
            detail=f"Coordinates ({x}, {y}) out of bounds. "
            f"Valid range: x=[0, {rx-1}], y=[0, {ry-1}]",
        )

    # Extract diffraction pattern at (x, y)
    if _current_dataset.ndim == 4:
        pattern = _current_dataset[x, y, :, :]
    else:
        pattern = _current_dataset[x, :, :]

    # Process for display with log scale and contrast
    normalized = _process_image_for_display(
        pattern, log_scale, contrast_min, contrast_max
    )

    # Convert to PNG
    image = Image.fromarray(normalized, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return DiffractionPatternResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
        x=x,
        y=y,
    )
