"""Image processing utilities for the 4D-STEM API."""

import base64
import io

import numpy as np
from PIL import Image


def process_image_for_display(
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


def create_region_mask(
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


def image_to_base64_png(data: np.ndarray, mode: str = "L") -> str:
    """
    Convert a numpy array to a base64-encoded PNG string.

    Parameters
    ----------
    data : np.ndarray
        Image data (should be uint8 for grayscale).
    mode : str
        PIL image mode. Default is "L" for grayscale.

    Returns
    -------
    str
        Base64-encoded PNG image.
    """
    image = Image.fromarray(data, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
