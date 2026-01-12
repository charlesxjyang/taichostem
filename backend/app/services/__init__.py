"""Services for the 4D-STEM API."""

from app.services.state import (
    get_current_dataset,
    set_current_dataset,
    get_current_dataset_path,
    get_cached_mean_diffraction,
    get_cached_max_diffraction,
    get_current_virtual_image,
    set_current_virtual_image,
    get_calibration,
    update_calibration,
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
    compute_diffraction_stats,
)
from app.services.image_processing import (
    process_image_for_display,
    create_region_mask,
    image_to_base64_png,
)
from app.services.hdf5 import (
    walk_hdf5_tree,
    build_hdf5_tree,
)

__all__ = [
    # State
    "get_current_dataset",
    "set_current_dataset",
    "get_current_dataset_path",
    "get_cached_mean_diffraction",
    "get_cached_max_diffraction",
    "get_current_virtual_image",
    "set_current_virtual_image",
    "get_calibration",
    "update_calibration",
    "get_probe_template",
    "set_probe_template",
    "get_probe_template_position",
    "set_probe_template_position",
    "get_probe_calibration",
    "set_probe_calibration",
    "get_correlation_kernel",
    "set_correlation_kernel",
    "get_kernel_type",
    "set_kernel_type",
    "clear_probe_state",
    "compute_diffraction_stats",
    # Image processing
    "process_image_for_display",
    "create_region_mask",
    "image_to_base64_png",
    # HDF5
    "walk_hdf5_tree",
    "build_hdf5_tree",
]
