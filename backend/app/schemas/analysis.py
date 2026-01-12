"""Analysis-related Pydantic schemas (atom detection, disk detection, preprocessing)."""

from typing import Literal

from pydantic import BaseModel


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
    qx0: float | None = None  # Probe center x-coordinate
    qy0: float | None = None  # Probe center y-coordinate


class ProbeStatusResponse(BaseModel):
    """Response model for probe template status."""

    is_set: bool
    shape: list[int] | None = None
    max_intensity: float | None = None
    position: list[int] | None = None
    # Probe calibration parameters
    alpha: float | None = None  # Convergence semi-angle (radius) in pixels
    qx0: float | None = None  # Probe center x-coordinate
    qy0: float | None = None  # Probe center y-coordinate


class GenerateKernelRequest(BaseModel):
    """Request model for generating cross-correlation kernel."""

    kernel_type: str = "sigmoid"  # "sigmoid", "gaussian", or "raw"
    radial_boundary: float = 0.5  # Fraction of probe radius for sigmoid transition
    sigmoid_width: float = 0.1  # Width of sigmoid transition (fraction of radius)


class GenerateKernelResponse(BaseModel):
    """Response model for generated kernel."""

    kernel_preview: str | None  # Optional kernel preview (deprecated)
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
