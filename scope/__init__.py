"""SCOPE: Soil Canopy Observation, Photochemistry and Energy fluxes model.

Python implementation of the SCOPE radiative transfer and energy balance model.

Example usage:
    >>> from scope import Constants, SpectralBands, run_scope
    >>> from scope.types import LeafBio, Canopy, Soil, Meteo, Angles
    >>>
    >>> constants = Constants()
    >>> spectral = SpectralBands()
    >>>
    >>> leafbio = LeafBio(Cab=40.0, Vcmax25=60.0)
    >>> canopy = Canopy(LAI=3.0, hc=2.0)
"""

from ._version import __version__
from .constants import CONSTANTS, TEMP_RESPONSE, Constants, TemperatureResponseParams
from .spectral import SPECTRAL, SpectralBands, SpectralRegion
from .main import run_scope

# Import submodules for convenience
from . import rtm
from . import fluxes
from . import supporting
from . import types
from . import io

__all__ = [
    # Version
    "__version__",
    # Constants
    "Constants",
    "TemperatureResponseParams",
    "CONSTANTS",
    "TEMP_RESPONSE",
    # Spectral
    "SpectralBands",
    "SpectralRegion",
    "SPECTRAL",
    "run_scope",
    # Submodules
    "rtm",
    "fluxes",
    "supporting",
    "types",
    "io",
]
