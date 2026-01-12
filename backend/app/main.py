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
# Stores the current virtual image for atom detection
_current_virtual_image: np.ndarray | None = None
# Cached mean and max diffraction patterns (computed on load)
_cached_mean_diffraction: np.ndarray | None = None
_cached_max_diffraction: np.ndarray | None = None
# Calibration values for real and reciprocal space
_calibration: dict[str, float | str] = {
    "q_pixel_size": 1.0,
    "q_pixel_units": "1/Å",
    "r_pixel_size": 1.0,
    "r_pixel_units": "Å",
}
# Probe template for disk detection (from vacuum region)
_probe_template: np.ndarray | None = None
_probe_template_position: tuple[int, int] | None = None
# Cross-correlation kernel generated from probe template
_correlation_kernel: np.ndarray | None = None
_kernel_type: str | None = None
# Probe calibration parameters (from get_probe_size)
_probe_alpha: float | None = None  # Convergence semi-angle (radius) in pixels
_probe_qx0: float | None = None    # Probe center x-coordinate
_probe_qy0: float | None = None    # Probe center y-coordinate


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


class HDF5TreeNode(BaseModel):
    """A node in the HDF5 file tree (group or dataset)."""

    name: str
    type: Literal["group", "dataset"]
    path: str
    children: list["HDF5TreeNode"] | None = None
    # Dataset-specific fields (only present when type == "dataset")
    shape: list[int] | None = None
    dtype: str | None = None
    is_4d: bool | None = None


# Enable self-referencing for recursive model
HDF5TreeNode.model_rebuild()


class SingleProbeResponse(BaseModel):
    """Response for single-datacube files (.dm4, .mrc)."""

    type: Literal["single"]
    shape: list[int]
    dtype: str


class HDF5TreeProbeResponse(BaseModel):
    """Response for HDF5 files with a hierarchical tree structure."""

    type: Literal["hdf5_tree"]
    filename: str
    root: HDF5TreeNode
    # Keep flat list for backwards compatibility
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


class MaxDiffractionResponse(BaseModel):
    """Response model for max diffraction pattern."""

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


def _compute_diffraction_stats() -> None:
    """
    Compute and cache mean and max diffraction patterns from the current dataset.

    Uses py4DSTEM's get_dp_mean() and get_dp_max() methods for efficient computation.
    Should be called after loading a dataset. Updates the global cache variables
    _cached_mean_diffraction and _cached_max_diffraction.
    """
    global _cached_mean_diffraction, _cached_max_diffraction

    if _current_dataset is None:
        _cached_mean_diffraction = None
        _cached_max_diffraction = None
        return

    try:
        # Create py4DSTEM DataCube and use its methods
        datacube = py4DSTEM.DataCube(data=_current_dataset)

        # Get mean diffraction pattern
        mean_dp = datacube.get_dp_mean()
        if hasattr(mean_dp, "data"):
            _cached_mean_diffraction = mean_dp.data
        else:
            _cached_mean_diffraction = mean_dp

        # Get max diffraction pattern
        max_dp = datacube.get_dp_max()
        if hasattr(max_dp, "data"):
            _cached_max_diffraction = max_dp.data
        else:
            _cached_max_diffraction = max_dp

    except Exception:
        # Fallback to numpy if py4DSTEM methods fail
        if _current_dataset.ndim == 4:
            _cached_mean_diffraction = np.mean(_current_dataset, axis=(0, 1))
            _cached_max_diffraction = np.max(_current_dataset, axis=(0, 1))
        elif _current_dataset.ndim == 3:
            _cached_mean_diffraction = np.mean(_current_dataset, axis=0)
            _cached_max_diffraction = np.max(_current_dataset, axis=0)
        else:
            _cached_mean_diffraction = _current_dataset.copy()
            _cached_max_diffraction = _current_dataset.copy()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _walk_hdf5_tree(h5file: h5py.File) -> list[HDF5DatasetInfo]:
    """
    Recursively walk an HDF5 file and collect dataset information (flat list).

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


def _build_hdf5_tree(h5file: h5py.File) -> HDF5TreeNode:
    """
    Build a hierarchical tree structure from an HDF5 file.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    HDF5TreeNode
        Root node of the hierarchical tree structure.
    """

    def build_node(name: str, obj: h5py.HLObject, path: str) -> HDF5TreeNode:
        """Recursively build tree nodes."""
        if isinstance(obj, h5py.Group):
            children: list[HDF5TreeNode] = []
            for child_name in sorted(obj.keys()):
                try:
                    child_obj = obj[child_name]
                    child_path = f"{path}/{child_name}" if path != "/" else f"/{child_name}"
                    child_node = build_node(child_name, child_obj, child_path)
                    if child_node is not None:
                        children.append(child_node)
                except Exception:
                    # Skip items that can't be accessed
                    continue
            return HDF5TreeNode(
                name=name,
                type="group",
                path=path,
                children=children,
            )
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            # Only include datasets with 2 or more dimensions
            if len(shape) >= 2:
                return HDF5TreeNode(
                    name=name,
                    type="dataset",
                    path=path,
                    shape=list(shape),
                    dtype=str(obj.dtype),
                    is_4d=len(shape) == 4,
                )
            else:
                # Skip 1D or 0D datasets
                return None
        return None

    # Build the root node from the file's root group
    root_children: list[HDF5TreeNode] = []
    for name in sorted(h5file.keys()):
        try:
            obj = h5file[name]
            node = build_node(name, obj, f"/{name}")
            if node is not None:
                root_children.append(node)
        except Exception:
            continue

    return HDF5TreeNode(
        name="/",
        type="group",
        path="/",
        children=root_children,
    )


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
                tree = _build_hdf5_tree(h5file)
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

    global _current_dataset, _current_dataset_path, _cached_mean_diffraction, _cached_max_diffraction

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
                    _compute_diffraction_stats()
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
                        _compute_diffraction_stats()
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
                        _compute_diffraction_stats()
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
        _compute_diffraction_stats()

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
    if _cached_mean_diffraction is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Use cached mean pattern
    mean_pattern = _cached_mean_diffraction

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


@app.get("/dataset/diffraction/max", response_model=MaxDiffractionResponse)
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
    if _cached_max_diffraction is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Use cached max pattern
    max_pattern = _cached_max_diffraction

    # Process for display with log scale and contrast
    normalized = _process_image_for_display(
        max_pattern, log_scale, contrast_min, contrast_max
    )

    # Convert to PNG
    image = Image.fromarray(normalized, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return MaxDiffractionResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
    )


@app.get("/dataset/virtual-image", response_model=VirtualImageResponse)
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
    - ABF (annular bright-field): Annular region just outside BF disk,
      captures the region between the BF disk edge and the ADF inner radius.
      Useful for light element detection.
    - ADF (annular dark-field): Outer annular ring, uses both inner and outer radii

    Parameters
    ----------
    type : Literal["bf", "abf", "adf"]
        Detector type. "bf" for bright-field (disk), "abf" for annular bright-field,
        "adf" for annular dark-field.
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
        _, _, qx, qy = _current_dataset.shape
    elif _current_dataset.ndim == 3:
        _, qx, qy = _current_dataset.shape
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must be 3D or 4D, got {_current_dataset.ndim}D",
        )

    global _current_virtual_image

    try:
        # Create py4DSTEM DataCube from current dataset
        datacube = py4DSTEM.DataCube(data=_current_dataset)

        # Calculate center of diffraction pattern
        center_x, center_y = qx / 2, qy / 2

        # Use py4DSTEM's get_virtual_image method
        # py4DSTEM expects geometry as ((qx, qy), radius) for circle
        # and ((qx, qy), (inner_radius, outer_radius)) for annulus
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
        _current_virtual_image = virtual_image.copy()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute virtual image: {str(e)}",
        )

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


# --- Region Diffraction Schemas ---


class RegionDiffractionRequest(BaseModel):
    """Request model for region-based diffraction pattern.

    Supports three region types:
    - rectangle: points contains [[x1,y1], [x2,y2]] (opposite corners)
    - ellipse: points contains [[centerX, centerY], [edgeX, edgeY]]
      where distance from center to edge defines semi-axes
    - polygon: points contains array of [x,y] vertices
    """

    mode: Literal["mean", "max"]
    region_type: Literal["rectangle", "ellipse", "polygon"]
    points: list[list[float]]
    log_scale: bool = False
    contrast_min: float = 0.0
    contrast_max: float = 100.0


class RegionDiffractionResponse(BaseModel):
    """Response model for region-based diffraction pattern."""

    image_base64: str
    width: int
    height: int
    mode: str
    region_type: str
    pixels_in_region: int


def _create_region_mask(
    region_type: str,
    points: list[list[float]],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Create a boolean mask for the specified region.

    Parameters
    ----------
    region_type : str
        Type of region: "rectangle", "ellipse", or "polygon".
    points : list[list[float]]
        Points defining the region:
        - rectangle: [[x1, y1], [x2, y2]] (opposite corners)
        - ellipse: [[centerX, centerY], [edgeX, edgeY]]
        - polygon: [[x1, y1], [x2, y2], ...] (vertices)
    shape : tuple[int, int]
        Shape of the output mask (height, width).

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates pixels inside the region.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)

    if region_type == "rectangle":
        if len(points) < 2:
            return mask
        x1, y1 = points[0]
        x2, y2 = points[1]
        # Ensure proper ordering
        x_min, x_max = int(min(x1, x2)), int(max(x1, x2))
        y_min, y_max = int(min(y1, y2)), int(max(y1, y2))
        # Clamp to image bounds
        x_min = max(0, x_min)
        x_max = min(width - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(height - 1, y_max)
        mask[y_min : y_max + 1, x_min : x_max + 1] = True

    elif region_type == "ellipse":
        if len(points) < 2:
            return mask
        center_x, center_y = points[0]
        edge_x, edge_y = points[1]
        # Calculate semi-axes from center to edge point
        semi_axis_x = abs(edge_x - center_x)
        semi_axis_y = abs(edge_y - center_y)
        if semi_axis_x < 0.5 and semi_axis_y < 0.5:
            return mask
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]
        # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
        # Handle case where one axis is very small
        if semi_axis_x < 0.5:
            semi_axis_x = 0.5
        if semi_axis_y < 0.5:
            semi_axis_y = 0.5
        ellipse_eq = (
            ((x_coords - center_x) / semi_axis_x) ** 2
            + ((y_coords - center_y) / semi_axis_y) ** 2
        )
        mask = ellipse_eq <= 1

    elif region_type == "polygon":
        if len(points) < 3:
            return mask
        try:
            from matplotlib.path import Path

            # Create a path from the polygon vertices
            vertices = np.array(points)
            path = Path(vertices)
            # Create coordinate grid
            x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
            coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
            # Check which points are inside
            inside = path.contains_points(coords)
            mask = inside.reshape((height, width))
        except ImportError:
            # Fallback: use a simple point-in-polygon algorithm
            # (This is slower but doesn't require matplotlib)
            from skimage.draw import polygon as draw_polygon

            vertices = np.array(points)
            rr, cc = draw_polygon(vertices[:, 1], vertices[:, 0], shape=shape)
            mask[rr, cc] = True

    return mask


@app.post("/dataset/diffraction/region", response_model=RegionDiffractionResponse)
async def get_region_diffraction(
    request: RegionDiffractionRequest,
) -> RegionDiffractionResponse:
    """
    Get mean or max diffraction pattern for a selected region.

    Computes the mean or max of diffraction patterns only for pixels
    within the specified region in real space. Uses py4DSTEM's
    get_virtual_diffraction() method for efficient computation, with
    a fallback to NumPy if py4DSTEM fails.

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
    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Get real space dimensions
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

    # Create mask for the region
    mask = _create_region_mask(request.region_type, request.points, (rx, ry))
    pixels_in_region = int(np.sum(mask))

    if pixels_in_region == 0:
        raise HTTPException(
            status_code=400,
            detail="Selected region contains no pixels",
        )

    # Compute diffraction pattern using py4DSTEM with fallback to NumPy
    try:
        # Create py4DSTEM DataCube and use get_virtual_diffraction
        datacube = py4DSTEM.DataCube(data=_current_dataset)
        
        # Use py4DSTEM's get_virtual_diffraction method with mask
        virtual_diffraction = datacube.get_virtual_diffraction(
            method=request.mode,  # 'mean' or 'max'
            mask=mask,
            name=f"{request.region_type}_{request.mode}_diffraction",
        )
        
        # Extract the data array from the DiffractionImage object
        if hasattr(virtual_diffraction, "data"):
            result = virtual_diffraction.data
        else:
            result = virtual_diffraction
    
    except Exception:
        # Fallback to NumPy if py4DSTEM method fails
        if _current_dataset.ndim == 4:
            # Get indices where mask is True
            indices = np.argwhere(mask)
            # Extract patterns at those positions
            patterns = np.array([_current_dataset[i, j, :, :] for i, j in indices])
        else:
            indices = np.argwhere(mask[:, 0])
            patterns = np.array([_current_dataset[i, :, :] for i in indices.flatten()])

        # Compute mean or max
        if request.mode == "mean":
            result = np.mean(patterns, axis=0)
        else:  # max
            result = np.max(patterns, axis=0)

    # Process for display
    normalized = _process_image_for_display(
        result, request.log_scale, request.contrast_min, request.contrast_max
    )

    # Convert to PNG
    image = Image.fromarray(normalized, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return RegionDiffractionResponse(
        image_base64=image_base64,
        width=normalized.shape[1],
        height=normalized.shape[0],
        mode=request.mode,
        region_type=request.region_type,
        pixels_in_region=pixels_in_region,
    )


# --- Atom Detection Schemas ---


class FindAtomsRequest(BaseModel):
    """Request model for atom detection."""

    threshold: float = 0.3
    min_distance: int = 5
    refine: bool = True


class FindAtomsResponse(BaseModel):
    """Response model for atom detection results."""

    count: int
    positions: list[list[float]]


# --- Preprocessing Schemas ---


class FilterHotPixelsRequest(BaseModel):
    """Request model for hot pixel filtering."""

    thresh: float = 8.0


class FilterHotPixelsResponse(BaseModel):
    """Response model for hot pixel filtering."""

    success: bool
    pixels_filtered: int
    message: str


# --- Disk Detection Schemas ---


class ExtractTemplateRequest(BaseModel):
    """Request model for extracting disk template from ROI."""

    x1: int
    y1: int
    x2: int
    y2: int


class ExtractTemplateResponse(BaseModel):
    """Response model for extracted template."""

    template_base64: str
    width: int
    height: int


class AutoDetectTemplateResponse(BaseModel):
    """Response model for auto-detected template."""

    template_base64: str
    width: int
    height: int
    x1: int
    y1: int
    x2: int
    y2: int


class SetProbeRequest(BaseModel):
    """Request model for setting probe template from a scan position or region.

    Supports two modes:
    1. Point selection: Provide x, y coordinates for a single scan position
    2. Region selection: Provide region_type and points to average over a region

    For vacuum probe extraction, a region is preferred as it averages out noise.
    """

    # Point selection (optional if region is provided)
    x: int | None = None
    y: int | None = None

    # Region selection (optional if x, y provided)
    region_type: Literal["rectangle", "ellipse", "polygon"] | None = None
    points: list[list[float]] | None = None


class SetProbeResponse(BaseModel):
    """Response model for set probe template."""

    shape: list[int]
    max_intensity: float
    center_x: float
    center_y: float
    preview: str
    position: list[int] | None  # [x, y] for point selection
    source_type: str  # "point" or "region"
    pixels_averaged: int  # 1 for point, N for region
    # Probe calibration parameters
    alpha: float | None = None  # Convergence semi-angle (radius) in pixels
    qx0: float | None = None    # Probe center x-coordinate  
    qy0: float | None = None    # Probe center y-coordinate


class ProbeStatusResponse(BaseModel):
    """Response model for probe template status."""

    is_set: bool
    shape: list[int] | None = None
    max_intensity: float | None = None
    position: list[int] | None = None
    # Probe calibration parameters
    alpha: float | None = None  # Convergence semi-angle (radius) in pixels
    qx0: float | None = None    # Probe center x-coordinate
    qy0: float | None = None    # Probe center y-coordinate


class GenerateKernelRequest(BaseModel):
    """Request model for generating cross-correlation kernel."""

    kernel_type: str = "sigmoid"  # "sigmoid", "gaussian", or "raw"
    radial_boundary: float = 0.5  # Fraction of probe radius for sigmoid transition
    sigmoid_width: float = 0.1  # Width of sigmoid transition (fraction of radius)


class GenerateKernelResponse(BaseModel):
    """Response model for generated kernel."""

    kernel_preview: str  # 50x50 grayscale kernel image
    kernel_lineprofile: str | None  # py4DSTEM show_kernel visualization
    kernel_shape: list[int]
    kernel_type: str
    radial_boundary: float
    sigmoid_width: float


class KernelStatusResponse(BaseModel):
    """Response model for kernel status."""

    is_set: bool
    kernel_type: str | None = None
    kernel_shape: list[int] | None = None


class DiskDetectionTestRequest(BaseModel):
    """Request model for testing disk detection on specific positions."""

    positions: list[list[int]]  # List of [x, y] scan positions to test
    correlation_threshold: float = 0.3  # Minimum correlation for detection
    min_spacing: int = 5  # Minimum pixel distance between detected disks
    subpixel: bool = True  # Enable subpixel refinement
    edge_boundary: int = 2  # Exclude detections within this many pixels of edge


class DetectedDisk(BaseModel):
    """Information about a single detected disk."""

    qx: float  # x position in diffraction space
    qy: float  # y position in diffraction space
    correlation: float  # correlation strength


class PositionTestResult(BaseModel):
    """Test result for a single scan position."""

    position: list[int]  # [x, y] scan position
    disks: list[DetectedDisk]  # Detected disks at this position
    pattern_overlay: str  # Base64 PNG of diffraction pattern with disk overlay
    disk_count: int  # Number of detected disks


class DiskDetectionTestResponse(BaseModel):
    """Response model for disk detection test."""

    results: list[PositionTestResult]
    correlation_histogram: str | None  # Base64 PNG of correlation distribution


# --- Calibration Schemas ---


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


@app.get("/dataset/calibration", response_model=CalibrationResponse)
async def get_calibration() -> CalibrationResponse:
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
    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    return CalibrationResponse(
        q_pixel_size=float(_calibration["q_pixel_size"]),
        q_pixel_units=str(_calibration["q_pixel_units"]),
        r_pixel_size=float(_calibration["r_pixel_size"]),
        r_pixel_units=str(_calibration["r_pixel_units"]),
    )


@app.post("/dataset/calibration", response_model=CalibrationResponse)
async def set_calibration(request: CalibrationUpdateRequest) -> CalibrationResponse:
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
    global _calibration

    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Update only provided fields
    if request.q_pixel_size is not None:
        if request.q_pixel_size <= 0:
            raise HTTPException(
                status_code=400,
                detail="q_pixel_size must be positive",
            )
        _calibration["q_pixel_size"] = request.q_pixel_size

    if request.q_pixel_units is not None:
        _calibration["q_pixel_units"] = request.q_pixel_units

    if request.r_pixel_size is not None:
        if request.r_pixel_size <= 0:
            raise HTTPException(
                status_code=400,
                detail="r_pixel_size must be positive",
            )
        _calibration["r_pixel_size"] = request.r_pixel_size

    if request.r_pixel_units is not None:
        _calibration["r_pixel_units"] = request.r_pixel_units

    return CalibrationResponse(
        q_pixel_size=float(_calibration["q_pixel_size"]),
        q_pixel_units=str(_calibration["q_pixel_units"]),
        r_pixel_size=float(_calibration["r_pixel_size"]),
        r_pixel_units=str(_calibration["r_pixel_units"]),
    )


@app.post("/analysis/find-atoms", response_model=FindAtomsResponse)
async def find_atoms(request: FindAtomsRequest) -> FindAtomsResponse:
    """
    Find atoms in the current virtual image using peak detection.

    Uses scikit-image peak_local_max for initial detection, with optional
    Gaussian refinement for sub-pixel accuracy.

    Parameters
    ----------
    request : FindAtomsRequest
        threshold : float
            Minimum intensity threshold as fraction of max (0-1).
        min_distance : int
            Minimum distance between peaks in pixels.
        refine : bool
            If True, refine positions using Gaussian fitting.

    Returns
    -------
    FindAtomsResponse
        count : int
            Number of atoms found.
        positions : list[list[float]]
            List of [x, y] positions for each atom.

    Raises
    ------
    HTTPException
        400 if no virtual image is available.
    """
    from skimage.feature import peak_local_max
    from scipy.ndimage import gaussian_filter

    if _current_virtual_image is None:
        raise HTTPException(
            status_code=400,
            detail="No virtual image available. Generate a virtual image first.",
        )

    # Normalize image to 0-1 range
    img = _current_virtual_image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)

    # Apply slight smoothing to reduce noise
    img_smoothed = gaussian_filter(img_normalized, sigma=1.0)

    # Find peaks using local maxima detection
    # threshold_abs is the minimum intensity value for a peak
    threshold_abs = request.threshold * img_smoothed.max()

    coordinates = peak_local_max(
        img_smoothed,
        min_distance=request.min_distance,
        threshold_abs=threshold_abs,
        exclude_border=True,
    )

    positions: list[list[float]] = []

    if len(coordinates) > 0:
        if request.refine:
            # Gaussian refinement for sub-pixel accuracy
            for coord in coordinates:
                y, x = coord
                # Extract small region around peak for refinement
                half_size = max(2, request.min_distance // 2)
                y_min = max(0, y - half_size)
                y_max = min(img_smoothed.shape[0], y + half_size + 1)
                x_min = max(0, x - half_size)
                x_max = min(img_smoothed.shape[1], x + half_size + 1)

                region = img_smoothed[y_min:y_max, x_min:x_max]

                # Compute centroid for sub-pixel refinement
                if region.sum() > 0:
                    yy, xx = np.mgrid[0:region.shape[0], 0:region.shape[1]]
                    total = region.sum()
                    refined_y = (yy * region).sum() / total + y_min
                    refined_x = (xx * region).sum() / total + x_min
                    positions.append([float(refined_x), float(refined_y)])
                else:
                    positions.append([float(x), float(y)])
        else:
            # Use integer coordinates directly
            for coord in coordinates:
                y, x = coord
                positions.append([float(x), float(y)])

    return FindAtomsResponse(
        count=len(positions),
        positions=positions,
    )


@app.post("/dataset/preprocess/filter-hot-pixels", response_model=FilterHotPixelsResponse)
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
    global _current_dataset, _cached_mean_diffraction, _cached_max_diffraction
    global _current_virtual_image

    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    try:
        # Create a py4DSTEM DataCube from our numpy array
        datacube = py4DSTEM.DataCube(data=_current_dataset)

        # Count pixels before filtering (estimate hot pixels by threshold)
        mean_val = np.mean(_current_dataset)
        std_val = np.std(_current_dataset)
        hot_pixel_mask = _current_dataset > (mean_val + request.thresh * std_val)
        pixels_before = int(np.sum(hot_pixel_mask))

        # Apply hot pixel filter
        datacube.filter_hot_pixels(thresh=request.thresh)

        # Update the in-memory dataset with the filtered data
        _current_dataset = datacube.data

        # Recompute cached diffraction patterns
        _compute_diffraction_stats()

        # Clear virtual image cache (will be recomputed on next request)
        _current_virtual_image = None

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


@app.post("/analysis/disk-detection/extract-template", response_model=ExtractTemplateResponse)
async def extract_template(request: ExtractTemplateRequest) -> ExtractTemplateResponse:
    """
    Extract a disk template from a region of interest on the mean diffraction pattern.

    Parameters
    ----------
    request : ExtractTemplateRequest
        x1, y1, x2, y2 : int
            Coordinates defining the rectangular ROI.

    Returns
    -------
    ExtractTemplateResponse
        template_base64 : str
            Base64-encoded PNG of the extracted template.
        width, height : int
            Dimensions of the template in pixels.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded or ROI is invalid.
    """
    if _cached_mean_diffraction is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Ensure coordinates are in correct order
    x1, x2 = min(request.x1, request.x2), max(request.x1, request.x2)
    y1, y2 = min(request.y1, request.y2), max(request.y1, request.y2)

    # Validate bounds
    h, w = _cached_mean_diffraction.shape
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise HTTPException(
            status_code=400,
            detail=f"ROI out of bounds. Image size is {w}x{h}",
        )

    if x2 - x1 < 2 or y2 - y1 < 2:
        raise HTTPException(
            status_code=400,
            detail="ROI too small. Must be at least 2x2 pixels.",
        )

    # Extract the region
    template_data = _cached_mean_diffraction[y1:y2, x1:x2].copy()

    # Normalize to 0-255 for display
    template_min = template_data.min()
    template_max = template_data.max()
    if template_max > template_min:
        template_normalized = (
            (template_data - template_min) / (template_max - template_min) * 255
        ).astype(np.uint8)
    else:
        template_normalized = np.zeros_like(template_data, dtype=np.uint8)

    # Convert to PNG
    img = Image.fromarray(template_normalized, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    template_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return ExtractTemplateResponse(
        template_base64=template_base64,
        width=template_normalized.shape[1],
        height=template_normalized.shape[0],
    )


@app.post("/analysis/disk-detection/auto-detect-template", response_model=AutoDetectTemplateResponse)
async def auto_detect_template() -> AutoDetectTemplateResponse:
    """
    Automatically detect and extract the central beam disk from the mean diffraction pattern.

    Uses intensity thresholding and centroid detection to find the brightest region,
    then extracts a square region around it.

    Returns
    -------
    AutoDetectTemplateResponse
        template_base64 : str
            Base64-encoded PNG of the extracted template.
        width, height : int
            Dimensions of the template in pixels.
        x1, y1, x2, y2 : int
            Coordinates of the detected ROI.

    Raises
    ------
    HTTPException
        400 if no dataset is loaded.
        500 if auto-detection fails.
    """
    from scipy.ndimage import center_of_mass, label

    if _cached_mean_diffraction is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    try:
        mean_dp = _cached_mean_diffraction.astype(np.float64)
        h, w = mean_dp.shape

        # Find threshold for bright region (top 5% intensity)
        threshold = np.percentile(mean_dp, 95)

        # Create binary mask of bright regions
        bright_mask = mean_dp > threshold

        # Label connected components
        labeled, num_features = label(bright_mask)

        if num_features == 0:
            # Fallback: use center of image
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 8
        else:
            # Find the largest bright region (likely the central beam)
            region_sizes = [
                np.sum(labeled == i) for i in range(1, num_features + 1)
            ]
            largest_region = np.argmax(region_sizes) + 1

            # Get centroid of largest region
            region_mask = labeled == largest_region
            cy, cx = center_of_mass(mean_dp, labels=labeled, index=largest_region)
            cy, cx = int(round(cy)), int(round(cx))

            # Estimate radius from region area
            area = region_sizes[largest_region - 1]
            radius = int(np.sqrt(area / np.pi) * 1.5)  # Add margin
            radius = max(radius, 8)  # Minimum radius

        # Create square ROI around centroid
        half_size = radius
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)

        # Extract the region
        template_data = _cached_mean_diffraction[y1:y2, x1:x2].copy()

        # Normalize to 0-255 for display
        template_min = template_data.min()
        template_max = template_data.max()
        if template_max > template_min:
            template_normalized = (
                (template_data - template_min) / (template_max - template_min) * 255
            ).astype(np.uint8)
        else:
            template_normalized = np.zeros_like(template_data, dtype=np.uint8)

        # Convert to PNG
        img = Image.fromarray(template_normalized, mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        template_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return AutoDetectTemplateResponse(
            template_base64=template_base64,
            width=template_normalized.shape[1],
            height=template_normalized.shape[0],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Auto-detection failed: {str(e)}",
        )


@app.post("/analysis/disk-detection/set-probe", response_model=SetProbeResponse)
async def set_probe_template(request: SetProbeRequest) -> SetProbeResponse:
    """
    Set the probe template by extracting diffraction pattern(s) from a scan position or region.

    Uses py4DSTEM's get_vacuum_probe() for region selection (recommended) which properly
    aligns and averages patterns. Automatically measures probe parameters using get_probe_size().

    Parameters
    ----------
    request : SetProbeRequest
        Supports two modes:
        - Point: x, y coordinates for single position extraction
        - Region: region_type and points for averaging over a region (recommended)

    Returns
    -------
    SetProbeResponse
        Probe template metadata including calibrated parameters (alpha, qx0, qy0).
    """
    global _probe_template, _probe_template_position, _probe_alpha, _probe_qx0, _probe_qy0

    if _current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Get dataset dimensions
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

    # Determine selection mode
    has_point = request.x is not None and request.y is not None
    has_region = request.region_type is not None and request.points is not None

    if not has_point and not has_region:
        raise HTTPException(
            status_code=400,
            detail="Must provide either (x, y) coordinates or (region_type, points) for selection.",
        )

    # Process based on selection mode
    if has_region:
        # Region selection: Use py4DSTEM's get_vacuum_probe for proper alignment & averaging
        mask = _create_region_mask(request.region_type, request.points, (rx, ry))
        pixels_in_region = int(np.sum(mask))

        if pixels_in_region == 0:
            raise HTTPException(
                status_code=400,
                detail="Selected region contains no pixels.",
            )

        try:
            # Create py4DSTEM DataCube and use get_vacuum_probe
            datacube = py4DSTEM.DataCube(data=_current_dataset)
            
            # get_vacuum_probe aligns patterns before averaging (default align=True)
            probe_obj = datacube.get_vacuum_probe(ROI=mask, returncalc=True)
            
            # Extract probe array from Probe object
            if hasattr(probe_obj, 'probe'):
                probe = probe_obj.probe
            else:
                probe = probe_obj
                
        except Exception as e:
            # Fallback to manual averaging if py4DSTEM method fails
            if _current_dataset.ndim == 4:
                indices = np.argwhere(mask)
                patterns = np.array([_current_dataset[i, j, :, :] for i, j in indices])
                probe = np.mean(patterns, axis=0)
            else:
                indices = np.argwhere(mask[:, 0])
                patterns = np.array([_current_dataset[i, :, :] for i in indices.flatten()])
                probe = np.mean(patterns, axis=0)

        source_type = "region"
        position = None
        pixels_averaged = pixels_in_region
        _probe_template_position = None
    else:
        # Point selection: extract single diffraction pattern
        if request.x < 0 or request.x >= rx or request.y < 0 or request.y >= ry:
            raise HTTPException(
                status_code=400,
                detail=f"Coordinates ({request.x}, {request.y}) out of bounds. "
                f"Valid range: x=[0, {rx-1}], y=[0, {ry-1}]",
            )

        if _current_dataset.ndim == 4:
            probe = _current_dataset[request.x, request.y, :, :].copy()
        else:
            probe = _current_dataset[request.x, :, :].copy()

        source_type = "point"
        position = [request.x, request.y]
        pixels_averaged = 1
        _probe_template_position = (request.x, request.y)

    # Store as probe template
    _probe_template = probe.astype(np.float64)

    # Use py4DSTEM's get_probe_size to measure probe parameters
    try:
        alpha_pr, qx0_pr, qy0_pr = py4DSTEM.process.calibration.get_probe_size(_probe_template)
        _probe_alpha = float(alpha_pr)
        _probe_qx0 = float(qx0_pr)
        _probe_qy0 = float(qy0_pr)
        cx, cy = qx0_pr, qy0_pr  # Use measured center
    except Exception:
        # Fallback to scipy center of mass if py4DSTEM method fails
        from scipy.ndimage import center_of_mass
        _probe_alpha = None
        _probe_qx0 = None
        _probe_qy0 = None
        
        probe_norm = probe.astype(np.float64)
        probe_norm = probe_norm - probe_norm.min()
        if probe_norm.max() > 0:
            probe_norm = probe_norm / probe_norm.max()
        
        cy, cx = center_of_mass(probe_norm)
        if np.isnan(cx) or np.isnan(cy):
            cx, cy = qy / 2, qx / 2

    # Calculate probe statistics
    max_intensity = float(np.max(probe))

    # Create preview image
    normalized = _process_image_for_display(probe, log_scale=True, contrast_min=1, contrast_max=99)
    image = Image.fromarray(normalized, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    preview_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return SetProbeResponse(
        shape=[int(qx), int(qy)],
        max_intensity=max_intensity,
        center_x=float(cx),
        center_y=float(cy),
        preview=preview_base64,
        position=position,
        source_type=source_type,
        pixels_averaged=pixels_averaged,
        alpha=_probe_alpha,
        qx0=_probe_qx0,
        qy0=_probe_qy0,
    )


@app.get("/analysis/disk-detection/probe-status", response_model=ProbeStatusResponse)
async def get_probe_status() -> ProbeStatusResponse:
    """
    Get the current probe template status including calibrated parameters.

    Returns
    -------
    ProbeStatusResponse
        Probe status including alpha (convergence semi-angle), qx0, qy0 if measured.
    """
    if _probe_template is None:
        return ProbeStatusResponse(is_set=False)

    return ProbeStatusResponse(
        is_set=True,
        shape=list(_probe_template.shape),
        max_intensity=float(np.max(_probe_template)),
        position=list(_probe_template_position) if _probe_template_position else None,
        alpha=_probe_alpha,
        qx0=_probe_qx0,
        qy0=_probe_qy0,
    )


@app.delete("/analysis/disk-detection/probe")
async def clear_probe_template() -> dict[str, bool]:
    """
    Clear the current probe template and calibration parameters.

    Returns
    -------
    dict
        success : bool
            Always True if no error.
    """
    global _probe_template, _probe_template_position, _correlation_kernel, _kernel_type
    global _probe_alpha, _probe_qx0, _probe_qy0

    _probe_template = None
    _probe_template_position = None
    _correlation_kernel = None
    _kernel_type = None
    _probe_alpha = None
    _probe_qx0 = None
    _probe_qy0 = None

    return {"success": True}


def _generate_sigmoid_kernel(
    probe: np.ndarray,
    radial_boundary: float = 0.5,
    sigmoid_width: float = 0.1,
) -> np.ndarray:
    """
    Generate a sigmoid-filtered kernel from a probe template.

    This creates a kernel optimized for cross-correlation disk detection
    by applying a sigmoid filter that emphasizes the disk edges.

    Parameters
    ----------
    probe : np.ndarray
        The probe template (2D array).
    radial_boundary : float
        Fraction of probe radius where sigmoid transition occurs (0-1).
    sigmoid_width : float
        Width of sigmoid transition as fraction of radius.

    Returns
    -------
    np.ndarray
        The sigmoid-filtered kernel.
    """
    from scipy.ndimage import center_of_mass, gaussian_filter

    # Get probe dimensions and center
    h, w = probe.shape
    cy, cx = h / 2, w / 2

    # Estimate probe radius from the probe itself
    probe_norm = probe.astype(np.float64)
    probe_norm = probe_norm - probe_norm.min()
    if probe_norm.max() > 0:
        probe_norm = probe_norm / probe_norm.max()

    # Find center of mass
    com_y, com_x = center_of_mass(probe_norm)
    if np.isnan(com_x) or np.isnan(com_y):
        com_x, com_y = cx, cy

    # Estimate radius using threshold
    threshold = 0.1
    mask = probe_norm > threshold
    if np.any(mask):
        y_coords, x_coords = np.where(mask)
        distances = np.sqrt((y_coords - com_y) ** 2 + (x_coords - com_x) ** 2)
        estimated_radius = np.percentile(distances, 90)
    else:
        estimated_radius = min(h, w) / 4

    # Create radial distance array from center
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - com_x) ** 2 + (y - com_y) ** 2)

    # Calculate sigmoid transition radius
    r_boundary = radial_boundary * estimated_radius
    r_width = max(sigmoid_width * estimated_radius, 1.0)

    # Create sigmoid filter: 1 at center, 0 at edges
    # Sigmoid: 1 / (1 + exp((r - r_boundary) / r_width))
    sigmoid_filter = 1.0 / (1.0 + np.exp((r - r_boundary) / r_width))

    # Apply sigmoid filter to probe
    kernel = probe.astype(np.float64) * sigmoid_filter

    # Subtract mean to create zero-mean kernel (better for correlation)
    kernel = kernel - kernel.mean()

    # Normalize
    kernel_std = kernel.std()
    if kernel_std > 0:
        kernel = kernel / kernel_std

    return kernel


def _generate_gaussian_kernel(probe: np.ndarray) -> np.ndarray:
    """
    Generate a Gaussian-filtered kernel from a probe template.

    Subtracts a Gaussian-smoothed version of the probe to emphasize edges,
    similar to py4DSTEM's get_probe_kernel_subtrgaussian approach.

    Parameters
    ----------
    probe : np.ndarray
        The probe template (2D array).

    Returns
    -------
    np.ndarray
        The Gaussian-filtered kernel.
    """
    from scipy.ndimage import gaussian_filter

    probe_float = probe.astype(np.float64)

    # Estimate sigma based on probe size
    sigma = min(probe.shape) / 8

    # Subtract Gaussian-smoothed version to emphasize edges
    smoothed = gaussian_filter(probe_float, sigma=sigma)
    kernel = probe_float - smoothed

    # Subtract mean to create zero-mean kernel
    kernel = kernel - kernel.mean()

    # Normalize
    kernel_std = kernel.std()
    if kernel_std > 0:
        kernel = kernel / kernel_std

    return kernel


@app.post("/analysis/disk-detection/generate-kernel", response_model=GenerateKernelResponse)
async def generate_kernel(request: GenerateKernelRequest) -> GenerateKernelResponse:
    """
    Generate a cross-correlation kernel from the stored probe template.

    The kernel is used for template matching to find Bragg disks.
    Different kernel types optimize for different detection scenarios:
    - sigmoid: Emphasizes disk edges with smooth transition (recommended)
    - gaussian: Subtracts Gaussian-smoothed probe to enhance edges
    - raw: Uses probe directly (not recommended for noisy data)

    Parameters
    ----------
    request : GenerateKernelRequest
        kernel_type : str
            Type of kernel: "sigmoid", "gaussian", or "raw".
        radial_boundary : float
            For sigmoid kernel: fraction of probe radius for transition (0-1).
        sigmoid_width : float
            For sigmoid kernel: width of transition (fraction of radius).

    Returns
    -------
    GenerateKernelResponse
        kernel_preview : str
            Base64-encoded PNG preview of the kernel.
        kernel_shape : list[int]
            Shape of the kernel [height, width].
        kernel_type : str
            Type of kernel generated.
        radial_boundary : float
            Radial boundary parameter used.
        sigmoid_width : float
            Sigmoid width parameter used.

    Raises
    ------
    HTTPException
        400 if no probe template is set.
        400 if invalid kernel type specified.
    """
    global _correlation_kernel, _kernel_type

    if _probe_template is None:
        raise HTTPException(
            status_code=400,
            detail="No probe template set. Use POST /analysis/disk-detection/set-probe first.",
        )

    if request.kernel_type not in ("sigmoid", "gaussian", "raw"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kernel type: {request.kernel_type}. Must be 'sigmoid', 'gaussian', or 'raw'.",
        )

    # Generate kernel based on type
    if request.kernel_type == "sigmoid":
        kernel = _generate_sigmoid_kernel(
            _probe_template,
            radial_boundary=request.radial_boundary,
            sigmoid_width=request.sigmoid_width,
        )
    elif request.kernel_type == "gaussian":
        kernel = _generate_gaussian_kernel(_probe_template)
    else:  # raw
        kernel = _probe_template.astype(np.float64)
        kernel = kernel - kernel.mean()
        kernel_std = kernel.std()
        if kernel_std > 0:
            kernel = kernel / kernel_std

    # Store kernel
    _correlation_kernel = kernel
    _kernel_type = request.kernel_type

    # Create preview images
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # 1. Create kernel diffraction space image (50x50 display)
    kernel_display = kernel.copy()
    kernel_min = kernel_display.min()
    kernel_max = kernel_display.max()
    if kernel_max > kernel_min:
        kernel_display = ((kernel_display - kernel_min) / (kernel_max - kernel_min) * 255).astype(np.uint8)
    else:
        kernel_display = np.zeros_like(kernel_display, dtype=np.uint8)

    # Resize to 50x50 for display
    kernel_image = Image.fromarray(kernel_display, mode="L")
    kernel_image = kernel_image.resize((50, 50), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    kernel_image.save(buffer, format="PNG")
    preview_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 2. Create line profile visualization using py4DSTEM's show_kernel
    try:
        kernel_size = min(kernel.shape)
        R = kernel_size // 2 - 2  # Leave some margin

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        py4DSTEM.visualize.show_kernel(
            kernel,
            R=R,
            L=R,
            W=1,
            figax=(fig, ax),
            returnfig=False,
        )
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="PNG", dpi=100, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        buffer.seek(0)
        lineprofile_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        plt.close("all")
        lineprofile_base64 = None

    return GenerateKernelResponse(
        kernel_preview=preview_base64,
        kernel_lineprofile=lineprofile_base64,
        kernel_shape=list(kernel.shape),
        kernel_type=request.kernel_type,
        radial_boundary=request.radial_boundary,
        sigmoid_width=request.sigmoid_width,
    )


@app.get("/analysis/disk-detection/kernel-status", response_model=KernelStatusResponse)
async def get_kernel_status() -> KernelStatusResponse:
    """
    Get the current kernel status.

    Returns
    -------
    KernelStatusResponse
        is_set : bool
            Whether a kernel has been generated.
        kernel_type : str | None
            Type of kernel if set.
        kernel_shape : list[int] | None
            Shape of kernel if set.
    """
    if _correlation_kernel is None:
        return KernelStatusResponse(is_set=False)

    return KernelStatusResponse(
        is_set=True,
        kernel_type=_kernel_type,
        kernel_shape=list(_correlation_kernel.shape),
    )


@app.post("/analysis/disk-detection/test", response_model=DiskDetectionTestResponse)
async def test_disk_detection(request: DiskDetectionTestRequest) -> DiskDetectionTestResponse:
    """
    Test disk detection on specific scan positions.

    Uses the stored correlation kernel to detect Bragg disks at the specified
    scan positions. Returns results with overlay visualizations for parameter tuning.

    Parameters
    ----------
    request : DiskDetectionTestRequest
        positions : list[list[int]]
            List of [x, y] scan positions to test.
        correlation_threshold : float
            Minimum correlation value for detection (0-1).
        min_spacing : int
            Minimum pixel distance between detected peaks.
        subpixel : bool
            Whether to refine positions to subpixel accuracy.
        edge_boundary : int
            Exclude detections within this many pixels of pattern edge.

    Returns
    -------
    DiskDetectionTestResponse
        results : list[PositionTestResult]
            Detection results for each position.
        correlation_histogram : str | None
            Base64 PNG histogram of correlation values.

    Raises
    ------
    HTTPException
        400 if no dataset loaded or no kernel set.
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import correlate2d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if _current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    if _correlation_kernel is None:
        raise HTTPException(status_code=400, detail="No correlation kernel set.")

    results = []
    all_correlations = []

    for pos in request.positions:
        rx, ry = pos[0], pos[1]

        # Bounds check
        if rx < 0 or rx >= _current_dataset.shape[0] or ry < 0 or ry >= _current_dataset.shape[1]:
            continue

        # Get diffraction pattern at this position
        dp = _current_dataset[rx, ry].astype(np.float64)

        # Cross-correlate with kernel
        correlation = correlate2d(dp, _correlation_kernel, mode="same")

        # Normalize correlation
        corr_max = np.max(np.abs(correlation))
        if corr_max > 0:
            correlation = correlation / corr_max

        # Find local maxima above threshold
        # Apply edge boundary
        edge = request.edge_boundary
        correlation_masked = correlation.copy()
        if edge > 0:
            correlation_masked[:edge, :] = 0
            correlation_masked[-edge:, :] = 0
            correlation_masked[:, :edge] = 0
            correlation_masked[:, -edge:] = 0

        # Find peaks using maximum filter
        neighborhood_size = max(3, request.min_spacing)
        local_max = maximum_filter(correlation_masked, size=neighborhood_size)
        peaks = (correlation_masked == local_max) & (correlation_masked >= request.correlation_threshold)

        # Get peak coordinates and values
        peak_coords = np.argwhere(peaks)
        peak_values = correlation_masked[peaks]

        # Sort by correlation strength
        sort_idx = np.argsort(peak_values)[::-1]
        peak_coords = peak_coords[sort_idx]
        peak_values = peak_values[sort_idx]

        # Filter peaks by minimum spacing (keep strongest)
        filtered_coords = []
        filtered_values = []
        for coord, val in zip(peak_coords, peak_values):
            qy, qx = coord  # Note: argwhere returns [row, col] = [y, x]
            # Check distance to already accepted peaks
            too_close = False
            for fc in filtered_coords:
                dist = np.sqrt((qx - fc[0])**2 + (qy - fc[1])**2)
                if dist < request.min_spacing:
                    too_close = True
                    break
            if not too_close:
                filtered_coords.append([qx, qy])
                filtered_values.append(val)
                all_correlations.append(val)

        # Subpixel refinement using center of mass
        if request.subpixel and len(filtered_coords) > 0:
            refined_coords = []
            window = 2  # pixels around peak for refinement
            for (qx, qy), val in zip(filtered_coords, filtered_values):
                x0 = max(0, int(qx) - window)
                x1 = min(correlation.shape[1], int(qx) + window + 1)
                y0 = max(0, int(qy) - window)
                y1 = min(correlation.shape[0], int(qy) + window + 1)

                region = correlation[y0:y1, x0:x1]
                if region.size > 0 and region.max() > 0:
                    # Threshold region
                    region = np.maximum(region - request.correlation_threshold, 0)
                    total = region.sum()
                    if total > 0:
                        yy, xx = np.mgrid[y0:y1, x0:x1]
                        refined_qx = (xx * region).sum() / total
                        refined_qy = (yy * region).sum() / total
                        refined_coords.append([float(refined_qx), float(refined_qy)])
                    else:
                        refined_coords.append([float(qx), float(qy)])
                else:
                    refined_coords.append([float(qx), float(qy)])
            filtered_coords = refined_coords

        # Create detected disk objects
        disks = []
        for (qx, qy), val in zip(filtered_coords, filtered_values):
            disks.append(DetectedDisk(qx=qx, qy=qy, correlation=float(val)))

        # Create overlay visualization
        fig, ax = plt.subplots(figsize=(4, 4))

        # Display diffraction pattern
        dp_display = dp.copy()
        dp_display = np.log10(dp_display + 1)  # Log scale for visibility
        ax.imshow(dp_display, cmap="gray", origin="upper")

        # Overlay detected disks with color-coded circles
        for disk in disks:
            # Color based on correlation strength
            if disk.correlation >= 0.7:
                color = "#4ade80"  # Green - high
            elif disk.correlation >= 0.5:
                color = "#facc15"  # Yellow - medium
            else:
                color = "#f87171"  # Red - low

            circle = plt.Circle(
                (disk.qx, disk.qy),
                radius=3,
                fill=False,
                color=color,
                linewidth=1.5,
            )
            ax.add_patch(circle)

        ax.set_xlim(0, dp.shape[1])
        ax.set_ylim(dp.shape[0], 0)
        ax.axis("off")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="PNG", dpi=100, bbox_inches="tight", facecolor="#1a1a1a", pad_inches=0.02)
        plt.close(fig)
        buffer.seek(0)
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        results.append(PositionTestResult(
            position=pos,
            disks=disks,
            pattern_overlay=overlay_base64,
            disk_count=len(disks),
        ))

    # Create correlation histogram
    histogram_base64 = None
    if len(all_correlations) > 0:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.hist(all_correlations, bins=20, range=(0, 1), color="#0066cc", edgecolor="#333", alpha=0.8)
        ax.axvline(x=request.correlation_threshold, color="#f87171", linestyle="--", linewidth=1.5, label="Threshold")
        ax.set_xlabel("Correlation", fontsize=8, color="#ccc")
        ax.set_ylabel("Count", fontsize=8, color="#ccc")
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.legend(fontsize=7, facecolor="#252525", edgecolor="#333", labelcolor="#ccc")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="PNG", dpi=100, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        buffer.seek(0)
        histogram_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return DiskDetectionTestResponse(
        results=results,
        correlation_histogram=histogram_base64,
    )
