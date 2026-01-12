"""HDF5 file handling utilities for the 4D-STEM API."""

import h5py

from app.schemas.dataset import HDF5DatasetInfo, HDF5TreeNode


def walk_hdf5_tree(h5file: h5py.File) -> list[HDF5DatasetInfo]:
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


def build_hdf5_tree(h5file: h5py.File) -> HDF5TreeNode:
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

    def build_node(name: str, obj: h5py.HLObject, path: str) -> HDF5TreeNode | None:
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
