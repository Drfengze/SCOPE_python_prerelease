"""Meteorological data for SCOPE model.

Translated from MATLAB struct definitions.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Meteo:
    """Meteorological input data.

    This dataclass contains atmospheric conditions required for
    energy balance and photosynthesis calculations.

    Attributes:
        z: Measurement height above canopy [m]
        Rin: Incoming shortwave radiation [W m-2]
        Ta: Air temperature at measurement height [°C]
        Rli: Incoming longwave radiation [W m-2]
        p: Atmospheric pressure [hPa]
        ea: Atmospheric vapor pressure [hPa]
        u: Wind speed at measurement height [m s-1]
        Ca: Atmospheric CO2 concentration [ppm]
        Oa: Atmospheric O2 concentration [per mille]
    """

    # Measurement configuration
    z: float = 10.0             # Measurement height [m]

    # Radiation
    Rin: float = 600.0          # Incoming shortwave [W m-2]
    Rli: float = 300.0          # Incoming longwave [W m-2]

    # Temperature and humidity
    Ta: float = 20.0            # Air temperature [°C]
    ea: float = 15.0            # Vapor pressure [hPa]

    # Pressure and wind
    p: float = 970.0            # Atmospheric pressure [hPa]
    u: float = 2.0              # Wind speed [m s-1]

    # Gas concentrations
    Ca: float = 380.0           # CO2 concentration [ppm]
    Oa: float = 209.0           # O2 concentration [per mille]

    # Leaf-level micrometeorology (for biochemical model)
    # These are typically computed during energy balance but can be set directly
    Q: Optional[float] = None   # Absorbed PAR [µmol m-2 s-1]
    Cs: Optional[float] = None  # CO2 at leaf surface [ppm] (defaults to Ca)
    eb: Optional[float] = None  # Vapor pressure at leaf boundary [hPa] (defaults to ea)
    T: Optional[float] = None   # Leaf temperature [°C] (defaults to Ta)

    # Stability parameters
    L: Optional[float] = None   # Monin-Obukhov length [m] (None = neutral)

    def __post_init__(self) -> None:
        """Validate meteorological inputs after initialization."""
        if self.Rin < 0:
            raise ValueError(f"Rin must be non-negative, got {self.Rin}")
        if self.Rli < 0:
            raise ValueError(f"Rli must be non-negative, got {self.Rli}")
        if self.u < 0:
            raise ValueError(f"Wind speed must be non-negative, got {self.u}")
        if self.p <= 0:
            raise ValueError(f"Pressure must be positive, got {self.p}")
        if self.Ca <= 0:
            raise ValueError(f"CO2 concentration must be positive, got {self.Ca}")

    @property
    def Ta_K(self) -> float:
        """Air temperature in Kelvin."""
        return self.Ta + 273.15

    @property
    def p_Pa(self) -> float:
        """Atmospheric pressure in Pascals."""
        return self.p * 100.0

    @property
    def ea_Pa(self) -> float:
        """Vapor pressure in Pascals."""
        return self.ea * 100.0

    @property
    def Cs_eff(self) -> float:
        """Effective CO2 at leaf surface [ppm]. Returns Cs if set, else Ca."""
        return self.Cs if self.Cs is not None else self.Ca

    @property
    def eb_eff(self) -> float:
        """Effective vapor pressure at leaf boundary [hPa]. Returns eb if set, else ea."""
        return self.eb if self.eb is not None else self.ea

    @property
    def T_eff(self) -> float:
        """Effective leaf temperature [°C]. Returns T if set, else Ta."""
        return self.T if self.T is not None else self.Ta

    def copy(self, **kwargs) -> "Meteo":
        """Create a copy with optionally modified parameters.

        Args:
            **kwargs: Parameters to override in the copy.

        Returns:
            New Meteo instance with modified parameters.
        """
        from dataclasses import asdict

        params = asdict(self)
        params.update(kwargs)
        return Meteo(**params)
