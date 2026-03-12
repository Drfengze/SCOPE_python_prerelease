"""Soil parameters for SCOPE model.

Translated from MATLAB struct definitions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Soil:
    """Soil optical, thermal, and hydraulic parameters.

    This dataclass contains parameters describing soil reflectance,
    thermal properties, and moisture characteristics.

    Attributes:
        spectrum: Soil spectrum number (column index in soil file)
        rs_thermal: Broadband thermal reflectance [-]
        SMC: Soil moisture content [0-1]
        rss: Soil evaporation resistance [s m-1]
        rbs: Soil boundary layer resistance [s m-1]
        cs: Soil heat capacity [J kg-1 K-1]
        rhos: Soil bulk density [kg m-3]
        lambdas: Soil thermal conductivity [W m-1 K-1]
        BSMBrightness: BSM model brightness parameter [-]
        BSMlat: BSM model latitude parameter [-]
        BSMlon: BSM model longitude parameter [-]
        CSSOIL: Soil drag coefficient [-]
        Ts: Initial soil temperature [°C] (2 layers)
        Tsold: Soil temperature history [°C] (12x2 array)
    """

    # Optical properties
    spectrum: int = 1                   # Soil spectrum number
    rs_thermal: float = 0.06            # Thermal reflectance [-]

    # Moisture
    SMC: float = 0.25                   # Soil moisture content [0-1]

    # Aerodynamic resistances
    rss: float = 500.0                  # Evaporation resistance [s m-1]
    rbs: float = 10.0                   # Boundary layer resistance [s m-1]
    CSSOIL: float = 0.01                # Soil drag coefficient [-]

    # Thermal properties
    cs: float = 1180.0                  # Heat capacity [J kg-1 K-1]
    rhos: float = 1800.0                # Bulk density [kg m-3]
    lambdas: float = 1.55               # Thermal conductivity [W m-1 K-1]

    # BSM model parameters (for simulating soil reflectance)
    BSMBrightness: float = 0.5          # Brightness [-]
    BSMlat: float = 25.0                # Latitude parameter [-]
    BSMlon: float = 45.0                # Longitude parameter [-]

    # Temperature state variables (initialized in __post_init__)
    _Ts: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _Tsold: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _refl: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize temperature arrays after dataclass creation."""
        # Initial soil surface temperature [°C] (2 layers: surface, deep)
        if self._Ts is None:
            self._Ts = np.array([15.0, 15.0], dtype=np.float64)

        # Soil temperature history (12 time steps × 2 layers)
        # Used for force-restore method
        if self._Tsold is None:
            self._Tsold = np.full((12, 2), 20.0, dtype=np.float64)

    @property
    def Ts(self) -> NDArray[np.float64]:
        """Soil surface temperature [°C] (2 layers)."""
        return self._Ts

    @Ts.setter
    def Ts(self, value: NDArray[np.float64]) -> None:
        """Set soil surface temperature."""
        self._Ts = np.asarray(value, dtype=np.float64)

    @property
    def Tsold(self) -> NDArray[np.float64]:
        """Soil temperature history [°C] (12×2 array)."""
        return self._Tsold

    @Tsold.setter
    def Tsold(self, value: NDArray[np.float64]) -> None:
        """Set soil temperature history."""
        self._Tsold = np.asarray(value, dtype=np.float64)

    @property
    def refl(self) -> Optional[NDArray[np.float64]]:
        """Soil spectral reflectance (full spectrum)."""
        return self._refl

    @refl.setter
    def refl(self, value: NDArray[np.float64]) -> None:
        """Set soil spectral reflectance."""
        self._refl = np.asarray(value, dtype=np.float64)

    @property
    def emis(self) -> float:
        """Soil thermal emissivity [-]."""
        return 1.0 - self.rs_thermal

    @property
    def GAM(self) -> float:
        """Soil thermal inertia [J m-2 s-0.5 K-1].

        GAM = sqrt(rhos * cs * lambdas)
        """
        return np.sqrt(self.rhos * self.cs * self.lambdas)

    def copy(self, **kwargs) -> "Soil":
        """Create a copy with optionally modified parameters.

        Args:
            **kwargs: Parameters to override in the copy.

        Returns:
            New Soil instance with modified parameters.
        """
        from dataclasses import asdict, fields

        # Get only the public fields
        params = {f.name: getattr(self, f.name)
                  for f in fields(self)
                  if not f.name.startswith("_")}
        params.update(kwargs)
        return Soil(**params)
