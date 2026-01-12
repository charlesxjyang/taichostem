"""Global state management for the 4D-STEM API.

This module manages the in-memory state for the currently loaded dataset,
cached computations, and analysis state (probe templates, kernels, etc.).
"""

import numpy as np
import py4DSTEM


# --- In-memory dataset cache ---
_current_dataset: np.ndarray | None = None
_current_dataset_path: str | None = None
_current_virtual_image: np.ndarray | None = None
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
_probe_qx0: float | None = None  # Probe center x-coordinate
_probe_qy0: float | None = None  # Probe center y-coordinate


# --- Dataset accessors ---


def get_current_dataset() -> np.ndarray | None:
    """Get the currently loaded dataset."""
    return _current_dataset


def set_current_dataset(dataset: np.ndarray | None, path: str | None = None) -> None:
    """Set the current dataset and optionally its path."""
    global _current_dataset, _current_dataset_path
    _current_dataset = dataset
    _current_dataset_path = path


def get_current_dataset_path() -> str | None:
    """Get the path of the currently loaded dataset."""
    return _current_dataset_path


def get_cached_mean_diffraction() -> np.ndarray | None:
    """Get the cached mean diffraction pattern."""
    return _cached_mean_diffraction


def get_cached_max_diffraction() -> np.ndarray | None:
    """Get the cached max diffraction pattern."""
    return _cached_max_diffraction


def get_current_virtual_image() -> np.ndarray | None:
    """Get the current virtual image."""
    return _current_virtual_image


def set_current_virtual_image(image: np.ndarray | None) -> None:
    """Set the current virtual image."""
    global _current_virtual_image
    _current_virtual_image = image


# --- Calibration accessors ---


def get_calibration() -> dict[str, float | str]:
    """Get the current calibration values."""
    return _calibration.copy()


def update_calibration(
    q_pixel_size: float | None = None,
    q_pixel_units: str | None = None,
    r_pixel_size: float | None = None,
    r_pixel_units: str | None = None,
) -> dict[str, float | str]:
    """
    Update calibration values.

    Only updates the provided fields; others remain unchanged.

    Parameters
    ----------
    q_pixel_size : float | None
        Reciprocal space pixel size.
    q_pixel_units : str | None
        Reciprocal space units.
    r_pixel_size : float | None
        Real space pixel size.
    r_pixel_units : str | None
        Real space units.

    Returns
    -------
    dict[str, float | str]
        Updated calibration values.
    """
    global _calibration

    if q_pixel_size is not None:
        _calibration["q_pixel_size"] = q_pixel_size
    if q_pixel_units is not None:
        _calibration["q_pixel_units"] = q_pixel_units
    if r_pixel_size is not None:
        _calibration["r_pixel_size"] = r_pixel_size
    if r_pixel_units is not None:
        _calibration["r_pixel_units"] = r_pixel_units

    return _calibration.copy()


# --- Probe template accessors ---


def get_probe_template() -> np.ndarray | None:
    """Get the current probe template."""
    return _probe_template


def set_probe_template(template: np.ndarray | None) -> None:
    """Set the probe template."""
    global _probe_template
    _probe_template = template


def get_probe_template_position() -> tuple[int, int] | None:
    """Get the position where the probe template was extracted."""
    return _probe_template_position


def set_probe_template_position(position: tuple[int, int] | None) -> None:
    """Set the probe template extraction position."""
    global _probe_template_position
    _probe_template_position = position


def get_probe_calibration() -> tuple[float | None, float | None, float | None]:
    """Get probe calibration parameters (alpha, qx0, qy0)."""
    return _probe_alpha, _probe_qx0, _probe_qy0


def set_probe_calibration(
    alpha: float | None,
    qx0: float | None,
    qy0: float | None,
) -> None:
    """Set probe calibration parameters."""
    global _probe_alpha, _probe_qx0, _probe_qy0
    _probe_alpha = alpha
    _probe_qx0 = qx0
    _probe_qy0 = qy0


# --- Kernel accessors ---


def get_correlation_kernel() -> np.ndarray | None:
    """Get the current correlation kernel."""
    return _correlation_kernel


def set_correlation_kernel(kernel: np.ndarray | None) -> None:
    """Set the correlation kernel."""
    global _correlation_kernel
    _correlation_kernel = kernel


def get_kernel_type() -> str | None:
    """Get the type of the current kernel."""
    return _kernel_type


def set_kernel_type(kernel_type: str | None) -> None:
    """Set the kernel type."""
    global _kernel_type
    _kernel_type = kernel_type


def clear_probe_state() -> None:
    """Clear all probe-related state (template, kernel, calibration)."""
    global _probe_template, _probe_template_position
    global _correlation_kernel, _kernel_type
    global _probe_alpha, _probe_qx0, _probe_qy0

    _probe_template = None
    _probe_template_position = None
    _correlation_kernel = None
    _kernel_type = None
    _probe_alpha = None
    _probe_qx0 = None
    _probe_qy0 = None


# --- Diffraction stats computation ---


def compute_diffraction_stats() -> None:
    """
    Compute and cache mean and max diffraction patterns from the current dataset.

    Uses py4DSTEM's get_dp_mean() and get_dp_max() methods for efficient computation.
    Should be called after loading a dataset. Updates the global cache variables.
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
