"""Analysis endpoints (atom detection, disk detection)."""

import base64
import io

import numpy as np
import py4DSTEM
from fastapi import APIRouter, HTTPException
from PIL import Image
from scipy.ndimage import gaussian_filter, center_of_mass, label, maximum_filter
from scipy.signal import correlate2d
from skimage.feature import peak_local_max

from app.schemas.analysis import (
    FindAtomsRequest,
    FindAtomsResponse,
    ExtractTemplateRequest,
    ExtractTemplateResponse,
    AutoDetectTemplateResponse,
    SetProbeRequest,
    SetProbeResponse,
    ProbeStatusResponse,
    GenerateKernelRequest,
    GenerateKernelResponse,
    KernelStatusResponse,
    DiskDetectionTestRequest,
    DiskDetectionTestResponse,
    DetectedDisk,
    PositionTestResult,
)
from app.services.state import (
    get_current_dataset,
    get_cached_mean_diffraction,
    get_current_virtual_image,
    get_probe_template,
    set_probe_template,
    get_probe_template_position,
    set_probe_template_position,
    get_probe_calibration,
    set_probe_calibration,
    get_correlation_kernel,
    set_correlation_kernel,
    get_kernel_type,
    set_kernel_type,
    clear_probe_state,
)
from app.services.image_processing import (
    process_image_for_display,
    create_region_mask,
    image_to_base64_png,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


# --- Atom Detection ---


@router.post("/find-atoms", response_model=FindAtomsResponse)
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
    current_virtual_image = get_current_virtual_image()
    if current_virtual_image is None:
        raise HTTPException(
            status_code=400,
            detail="No virtual image available. Generate a virtual image first.",
        )

    # Normalize image to 0-1 range
    img = current_virtual_image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img)

    # Apply slight smoothing to reduce noise
    img_smoothed = gaussian_filter(img_normalized, sigma=1.0)

    # Find peaks using local maxima detection
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
                half_size = max(2, request.min_distance // 2)
                y_min = max(0, y - half_size)
                y_max = min(img_smoothed.shape[0], y + half_size + 1)
                x_min = max(0, x - half_size)
                x_max = min(img_smoothed.shape[1], x + half_size + 1)

                region = img_smoothed[y_min:y_max, x_min:x_max]

                if region.sum() > 0:
                    yy, xx = np.mgrid[0 : region.shape[0], 0 : region.shape[1]]
                    total = region.sum()
                    refined_y = (yy * region).sum() / total + y_min
                    refined_x = (xx * region).sum() / total + x_min
                    positions.append([float(refined_x), float(refined_y)])
                else:
                    positions.append([float(x), float(y)])
        else:
            for coord in coordinates:
                y, x = coord
                positions.append([float(x), float(y)])

    return FindAtomsResponse(
        count=len(positions),
        positions=positions,
    )


# --- Disk Detection ---


@router.post("/disk-detection/extract-template", response_model=ExtractTemplateResponse)
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
    cached_mean = get_cached_mean_diffraction()
    if cached_mean is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Ensure coordinates are in correct order
    x1, x2 = min(request.x1, request.x2), max(request.x1, request.x2)
    y1, y2 = min(request.y1, request.y2), max(request.y1, request.y2)

    # Validate bounds
    h, w = cached_mean.shape
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
    template_data = cached_mean[y1:y2, x1:x2].copy()

    # Normalize to 0-255 for display
    template_min = template_data.min()
    template_max = template_data.max()
    if template_max > template_min:
        template_normalized = (
            (template_data - template_min) / (template_max - template_min) * 255
        ).astype(np.uint8)
    else:
        template_normalized = np.zeros_like(template_data, dtype=np.uint8)

    template_base64 = image_to_base64_png(template_normalized)

    return ExtractTemplateResponse(
        template_base64=template_base64,
        width=template_normalized.shape[1],
        height=template_normalized.shape[0],
    )


@router.post("/disk-detection/auto-detect-template", response_model=AutoDetectTemplateResponse)
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
    cached_mean = get_cached_mean_diffraction()
    if cached_mean is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    try:
        mean_dp = cached_mean.astype(np.float64)
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
            region_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_region = np.argmax(region_sizes) + 1

            # Get centroid of largest region
            region_mask = labeled == largest_region
            cy, cx = center_of_mass(mean_dp, labels=labeled, index=largest_region)
            cy, cx = int(round(cy)), int(round(cx))

            # Estimate radius from region area
            area = region_sizes[largest_region - 1]
            radius = int(np.sqrt(area / np.pi) * 1.5)
            radius = max(radius, 8)

        # Create square ROI around centroid
        half_size = radius
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)

        # Extract the region
        template_data = cached_mean[y1:y2, x1:x2].copy()

        # Normalize to 0-255 for display
        template_min = template_data.min()
        template_max = template_data.max()
        if template_max > template_min:
            template_normalized = (
                (template_data - template_min) / (template_max - template_min) * 255
            ).astype(np.uint8)
        else:
            template_normalized = np.zeros_like(template_data, dtype=np.uint8)

        template_base64 = image_to_base64_png(template_normalized)

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


@router.post("/disk-detection/set-probe", response_model=SetProbeResponse)
async def set_probe_template_endpoint(request: SetProbeRequest) -> SetProbeResponse:
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
    current_dataset = get_current_dataset()
    if current_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Load a dataset first using POST /dataset/load",
        )

    # Get dataset dimensions
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
        mask = create_region_mask(request.region_type, request.points, (rx, ry))
        pixels_in_region = int(np.sum(mask))

        if pixels_in_region == 0:
            raise HTTPException(
                status_code=400,
                detail="Selected region contains no pixels.",
            )

        try:
            datacube = py4DSTEM.DataCube(data=current_dataset)
            probe_obj = datacube.get_vacuum_probe(ROI=mask, returncalc=True)

            if hasattr(probe_obj, "probe"):
                probe = probe_obj.probe
            else:
                probe = probe_obj

        except Exception:
            if current_dataset.ndim == 4:
                indices = np.argwhere(mask)
                patterns = np.array([current_dataset[i, j, :, :] for i, j in indices])
                probe = np.mean(patterns, axis=0)
            else:
                indices = np.argwhere(mask[:, 0])
                patterns = np.array([current_dataset[i, :, :] for i in indices.flatten()])
                probe = np.mean(patterns, axis=0)

        source_type = "region"
        position = None
        pixels_averaged = pixels_in_region
        set_probe_template_position(None)
    else:
        if request.x < 0 or request.x >= rx or request.y < 0 or request.y >= ry:
            raise HTTPException(
                status_code=400,
                detail=f"Coordinates ({request.x}, {request.y}) out of bounds. "
                f"Valid range: x=[0, {rx-1}], y=[0, {ry-1}]",
            )

        if current_dataset.ndim == 4:
            probe = current_dataset[request.x, request.y, :, :].copy()
        else:
            probe = current_dataset[request.x, :, :].copy()

        source_type = "point"
        position = [request.x, request.y]
        pixels_averaged = 1
        set_probe_template_position((request.x, request.y))

    # Store as probe template
    set_probe_template(probe.astype(np.float64))

    # Use py4DSTEM's get_probe_size to measure probe parameters
    try:
        alpha_pr, qx0_pr, qy0_pr = py4DSTEM.process.calibration.get_probe_size(probe)
        set_probe_calibration(float(alpha_pr), float(qx0_pr), float(qy0_pr))
        cx, cy = qx0_pr, qy0_pr
    except Exception:
        set_probe_calibration(None, None, None)

        probe_norm = probe.astype(np.float64)
        probe_norm = probe_norm - probe_norm.min()
        if probe_norm.max() > 0:
            probe_norm = probe_norm / probe_norm.max()

        cy, cx = center_of_mass(probe_norm)
        if np.isnan(cx) or np.isnan(cy):
            cx, cy = qy / 2, qx / 2

    max_intensity = float(np.max(probe))

    # Create preview image
    normalized = process_image_for_display(probe, log_scale=True, contrast_min=1, contrast_max=99)
    preview_base64 = image_to_base64_png(normalized)

    alpha, qx0, qy0 = get_probe_calibration()

    return SetProbeResponse(
        shape=[int(qx), int(qy)],
        max_intensity=max_intensity,
        center_x=float(cx),
        center_y=float(cy),
        preview=preview_base64,
        position=position,
        source_type=source_type,
        pixels_averaged=pixels_averaged,
        alpha=alpha,
        qx0=qx0,
        qy0=qy0,
    )


@router.get("/disk-detection/probe-status", response_model=ProbeStatusResponse)
async def get_probe_status() -> ProbeStatusResponse:
    """
    Get the current probe template status including calibrated parameters.

    Returns
    -------
    ProbeStatusResponse
        Probe status including alpha (convergence semi-angle), qx0, qy0 if measured.
    """
    probe_template = get_probe_template()
    if probe_template is None:
        return ProbeStatusResponse(is_set=False)

    probe_position = get_probe_template_position()
    alpha, qx0, qy0 = get_probe_calibration()

    return ProbeStatusResponse(
        is_set=True,
        shape=list(probe_template.shape),
        max_intensity=float(np.max(probe_template)),
        position=list(probe_position) if probe_position else None,
        alpha=alpha,
        qx0=qx0,
        qy0=qy0,
    )


@router.delete("/disk-detection/probe")
async def clear_probe_template_endpoint() -> dict[str, bool]:
    """
    Clear the current probe template and calibration parameters.

    Returns
    -------
    dict
        success : bool
            Always True if no error.
    """
    clear_probe_state()
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
    h, w = probe.shape
    cy, cx = h / 2, w / 2

    probe_norm = probe.astype(np.float64)
    probe_norm = probe_norm - probe_norm.min()
    if probe_norm.max() > 0:
        probe_norm = probe_norm / probe_norm.max()

    com_y, com_x = center_of_mass(probe_norm)
    if np.isnan(com_x) or np.isnan(com_y):
        com_x, com_y = cx, cy

    threshold = 0.1
    mask = probe_norm > threshold
    if np.any(mask):
        y_coords, x_coords = np.where(mask)
        distances = np.sqrt((y_coords - com_y) ** 2 + (x_coords - com_x) ** 2)
        estimated_radius = np.percentile(distances, 90)
    else:
        estimated_radius = min(h, w) / 4

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - com_x) ** 2 + (y - com_y) ** 2)

    r_boundary = radial_boundary * estimated_radius
    r_width = max(sigmoid_width * estimated_radius, 1.0)

    sigmoid_filter = 1.0 / (1.0 + np.exp((r - r_boundary) / r_width))
    kernel = probe.astype(np.float64) * sigmoid_filter
    kernel = kernel - kernel.mean()

    kernel_std = kernel.std()
    if kernel_std > 0:
        kernel = kernel / kernel_std

    return kernel


def _generate_gaussian_kernel(probe: np.ndarray) -> np.ndarray:
    """
    Generate a Gaussian-filtered kernel from a probe template.

    Subtracts a Gaussian-smoothed version of the probe to emphasize edges.

    Parameters
    ----------
    probe : np.ndarray
        The probe template (2D array).

    Returns
    -------
    np.ndarray
        The Gaussian-filtered kernel.
    """
    probe_float = probe.astype(np.float64)
    sigma = min(probe.shape) / 8

    smoothed = gaussian_filter(probe_float, sigma=sigma)
    kernel = probe_float - smoothed
    kernel = kernel - kernel.mean()

    kernel_std = kernel.std()
    if kernel_std > 0:
        kernel = kernel / kernel_std

    return kernel


@router.post("/disk-detection/generate-kernel", response_model=GenerateKernelResponse)
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
    probe_template = get_probe_template()
    if probe_template is None:
        raise HTTPException(
            status_code=400,
            detail="No probe template set. Use POST /analysis/disk-detection/set-probe first.",
        )

    if request.kernel_type not in ("sigmoid", "gaussian", "raw"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid kernel type: {request.kernel_type}. Must be 'sigmoid', 'gaussian', or 'raw'.",
        )

    if request.kernel_type == "sigmoid":
        kernel = _generate_sigmoid_kernel(
            probe_template,
            radial_boundary=request.radial_boundary,
            sigmoid_width=request.sigmoid_width,
        )
    elif request.kernel_type == "gaussian":
        kernel = _generate_gaussian_kernel(probe_template)
    else:
        kernel = probe_template.astype(np.float64)
        kernel = kernel - kernel.mean()
        kernel_std = kernel.std()
        if kernel_std > 0:
            kernel = kernel / kernel_std

    set_correlation_kernel(kernel)
    set_kernel_type(request.kernel_type)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create line profile visualization
    try:
        # Fixed R value as specified
        R = 25
        
        # Get kernel center
        cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
        
        # Extract kernel region
        im_kernel = kernel[cy-R:cy+R, cx-R:cx+R]
        
        # Create two-panel figure
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
        
        # Left panel: 2D kernel visualization with crosshairs
        axs[0].matshow(im_kernel, cmap="gray")
        axs[0].plot(np.ones(2 * R) * R, np.arange(2 * R), c="r")  # Red vertical crosshair
        axs[0].plot(np.arange(2 * R), np.ones(2 * R) * R, c="c")  # Cyan horizontal crosshair
        
        # Right panel: Line profile plots
        lineprofile_1 = im_kernel[:, R]  # Vertical profile (y-axis)
        lineprofile_2 = im_kernel[R, :]  # Horizontal profile (x-axis)
        axs[1].plot(np.arange(len(lineprofile_1)), lineprofile_1, c="r")  # Red line
        axs[1].plot(np.arange(len(lineprofile_2)), lineprofile_2, c="c")  # Cyan line
        
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
        kernel_preview=None,  # Preview removed
        kernel_lineprofile=lineprofile_base64,
        kernel_shape=list(kernel.shape),
        kernel_type=request.kernel_type,
        radial_boundary=request.radial_boundary,
        sigmoid_width=request.sigmoid_width,
    )


@router.get("/disk-detection/kernel-status", response_model=KernelStatusResponse)
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
    correlation_kernel = get_correlation_kernel()
    if correlation_kernel is None:
        return KernelStatusResponse(is_set=False)

    return KernelStatusResponse(
        is_set=True,
        kernel_type=get_kernel_type(),
        kernel_shape=list(correlation_kernel.shape),
    )


@router.post("/disk-detection/test", response_model=DiskDetectionTestResponse)
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    current_dataset = get_current_dataset()
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")

    correlation_kernel = get_correlation_kernel()
    if correlation_kernel is None:
        raise HTTPException(status_code=400, detail="No correlation kernel set.")

    results = []
    all_correlations = []

    for pos in request.positions:
        rx, ry = pos[0], pos[1]

        if rx < 0 or rx >= current_dataset.shape[0] or ry < 0 or ry >= current_dataset.shape[1]:
            continue

        dp = current_dataset[rx, ry].astype(np.float64)
        correlation = correlate2d(dp, correlation_kernel, mode="same")

        corr_max = np.max(np.abs(correlation))
        if corr_max > 0:
            correlation = correlation / corr_max

        edge = request.edge_boundary
        correlation_masked = correlation.copy()
        if edge > 0:
            correlation_masked[:edge, :] = 0
            correlation_masked[-edge:, :] = 0
            correlation_masked[:, :edge] = 0
            correlation_masked[:, -edge:] = 0

        neighborhood_size = max(3, request.min_spacing)
        local_max = maximum_filter(correlation_masked, size=neighborhood_size)
        peaks = (correlation_masked == local_max) & (
            correlation_masked >= request.correlation_threshold
        )

        peak_coords = np.argwhere(peaks)
        peak_values = correlation_masked[peaks]

        sort_idx = np.argsort(peak_values)[::-1]
        peak_coords = peak_coords[sort_idx]
        peak_values = peak_values[sort_idx]

        filtered_coords = []
        filtered_values = []
        for coord, val in zip(peak_coords, peak_values):
            qy, qx = coord
            too_close = False
            for fc in filtered_coords:
                dist = np.sqrt((qx - fc[0]) ** 2 + (qy - fc[1]) ** 2)
                if dist < request.min_spacing:
                    too_close = True
                    break
            if not too_close:
                filtered_coords.append([qx, qy])
                filtered_values.append(val)
                all_correlations.append(val)

        if request.subpixel and len(filtered_coords) > 0:
            refined_coords = []
            window = 2
            for (qx, qy), val in zip(filtered_coords, filtered_values):
                x0 = max(0, int(qx) - window)
                x1 = min(correlation.shape[1], int(qx) + window + 1)
                y0 = max(0, int(qy) - window)
                y1 = min(correlation.shape[0], int(qy) + window + 1)

                region = correlation[y0:y1, x0:x1]
                if region.size > 0 and region.max() > 0:
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

        disks = []
        for (qx, qy), val in zip(filtered_coords, filtered_values):
            disks.append(DetectedDisk(qx=qx, qy=qy, correlation=float(val)))

        fig, ax = plt.subplots(figsize=(4, 4))
        dp_display = dp.copy()
        dp_display = np.log10(dp_display + 1)
        ax.imshow(dp_display, cmap="gray", origin="upper")

        for disk in disks:
            if disk.correlation >= 0.7:
                color = "#4ade80"
            elif disk.correlation >= 0.5:
                color = "#facc15"
            else:
                color = "#f87171"

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
        fig.savefig(
            buffer, format="PNG", dpi=100, bbox_inches="tight", facecolor="#1a1a1a", pad_inches=0.02
        )
        plt.close(fig)
        buffer.seek(0)
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        results.append(
            PositionTestResult(
                position=pos,
                disks=disks,
                pattern_overlay=overlay_base64,
                disk_count=len(disks),
            )
        )

    histogram_base64 = None
    if len(all_correlations) > 0:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.hist(
            all_correlations, bins=20, range=(0, 1), color="#0066cc", edgecolor="#333", alpha=0.8
        )
        ax.axvline(
            x=request.correlation_threshold,
            color="#f87171",
            linestyle="--",
            linewidth=1.5,
            label="Threshold",
        )
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
