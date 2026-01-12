"""Diffraction pattern and virtual image Pydantic schemas."""

from typing import Literal

from pydantic import BaseModel


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
