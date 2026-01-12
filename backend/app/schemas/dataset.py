"""Dataset-related Pydantic schemas."""

from typing import Literal

from pydantic import BaseModel


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


class DatasetLoadRequest(BaseModel):
    """Request model for loading a dataset."""

    path: str
    dataset_path: str | None = None


class DatasetLoadResponse(BaseModel):
    """Response model for loaded dataset metadata."""

    shape: list[int]
    dtype: str
    dataset_path: str | None = None
