"""Data structures for SCOPE model.

This module provides dataclass definitions for all input and output
structures used in SCOPE simulations.
"""

from .angles import Angles, LocationTime
from .canopy import Canopy
from .leafbio import LeafBio
from .meteo import Meteo
from .options import FilePaths, Options
from .radiation import (
    CanopyRadiation,
    FluorescenceOutput,
    LeafOptics,
    Radiation,
    ThermalOutput,
)
from .soil import Soil

__all__ = [
    # Input structures
    "Options",
    "FilePaths",
    "LeafBio",
    "Canopy",
    "Soil",
    "Meteo",
    "Angles",
    "LocationTime",
    # Output structures
    "LeafOptics",
    "CanopyRadiation",
    "FluorescenceOutput",
    "ThermalOutput",
    "Radiation",
]
