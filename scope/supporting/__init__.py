"""Supporting utility functions for SCOPE model.

This module contains utility functions used across SCOPE:
- Physics: Planck law, vapor pressure calculations
- Integration: Numerical integration routines
- LeafAngles: Leaf inclination distribution functions
- MeanLeaf: Leaf property averaging
"""

from .integration import cumulative_integral, sint, spectral_integral
from .leafangles import (
    campbell_lidf,
    compute_canopy_lidf,
    get_lidf_parameters,
    leafangles,
)
from .meanleaf import meanleaf, sunlit_shaded_average, weighted_layer_mean
from .physics import (
    e2phot,
    ephoton,
    phot2e,
    planck,
    relative_humidity,
    satvap,
    slope_satvap,
    stefan_boltzmann,
    vapor_pressure_deficit,
)

__all__ = [
    # Physics
    "ephoton",
    "e2phot",
    "phot2e",
    "satvap",
    "slope_satvap",
    "planck",
    "stefan_boltzmann",
    "vapor_pressure_deficit",
    "relative_humidity",
    # Integration
    "sint",
    "cumulative_integral",
    "spectral_integral",
    # Leaf angles
    "leafangles",
    "campbell_lidf",
    "get_lidf_parameters",
    "compute_canopy_lidf",
    # Mean leaf
    "meanleaf",
    "weighted_layer_mean",
    "sunlit_shaded_average",
]
