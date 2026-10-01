"""Multiscale representation: wafers, procedural microstructure, ROI extraction."""

from .region import AtomisticRegion, RegionSpec, extract_region, format_span, scale_ladder
from .wafer import SCALE_LEVELS, STANDARD_WAFERS, DeviceRegion, Layer, Wafer, WaferSpec

__all__ = [
    "Wafer", "WaferSpec", "Layer", "DeviceRegion", "STANDARD_WAFERS", "SCALE_LEVELS",
    "AtomisticRegion", "RegionSpec", "extract_region", "scale_ladder", "format_span",
]
