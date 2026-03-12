"""Canopy structure parameters for SCOPE model.

Translated from MATLAB struct definitions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Canopy:
    """Canopy structure and geometry parameters.

    This dataclass contains parameters describing canopy geometry,
    leaf angle distribution, and aerodynamic properties.

    Attributes:
        LAI: Leaf area index [m2 m-2]
        hc: Canopy height [m]
        LIDFa: Leaf inclination distribution function parameter a [-]
        LIDFb: Leaf inclination distribution function parameter b [-]
        leafwidth: Mean leaf width [m]
        kV: Vcmax vertical extinction coefficient [-] (default 0.64)
        rb: Leaf boundary layer resistance [s m-1]
        Cd: Leaf drag coefficient [-]
        CR: Drag coefficient for isolated tree [-]
        CD1: Roughness sublayer parameter [-]
        Psicor: Roughness layer correction factor [-]
        rwc: Within-canopy aerodynamic resistance [s m-1]
        zo: Roughness length for momentum [m]
        d: Zero-plane displacement height [m]
        nlayers: Number of canopy layers [-]
        nlincl: Number of leaf inclination classes [-]
        nlazi: Number of leaf azimuth classes [-]
    """

    # Vegetation parameters
    LAI: float = 3.0            # Leaf area index [m2 m-2]
    hc: float = 2.0             # Canopy height [m]
    LIDFa: float = -0.35        # LIDF parameter a [-]
    LIDFb: float = -0.15        # LIDF parameter b [-]
    leafwidth: float = 0.1      # Leaf width [m]
    kV: float = 0.6396          # Vcmax vertical extinction coefficient [-] (MATLAB default)

    # Aerodynamic parameters
    rb: float = 10.0            # Leaf boundary layer resistance [s m-1]
    Cd: float = 0.3             # Leaf drag coefficient [-]
    CR: float = 0.35            # Drag coefficient for isolated tree [-]
    CD1: float = 20.6           # Roughness sublayer parameter [-]
    Psicor: float = 0.2         # Roughness layer correction [-]
    rwc: float = 0.0            # Within-canopy resistance [s m-1]

    # Roughness parameters (can be calculated or specified)
    zo: Optional[float] = None  # Roughness length [m]
    d: Optional[float] = None   # Displacement height [m]

    # Layer discretization (fixed values from MATLAB)
    nlincl: int = 13            # Number of leaf inclination classes
    nlazi: int = 36             # Number of leaf azimuth classes

    # Internal arrays (computed in __post_init__)
    _nlayers: Optional[int] = field(default=None, repr=False)
    _litab: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _lazitab: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _x: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _xl: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _lidf: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize computed arrays after dataclass creation."""
        # Calculate number of layers (minimum 60, scales with LAI)
        if self._nlayers is None:
            self._nlayers = max(60, int(np.ceil(self.LAI * 20)))

        # Leaf inclination angles (SAIL convention)
        # 13 classes: 5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89 degrees
        self._litab = np.array([5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89],
                               dtype=np.float64)

        # Leaf azimuth angles (36 classes at 10° intervals)
        self._lazitab = np.arange(5, 360, 10, dtype=np.float64)

        # Cumulative LAI levels (x = 0 at top, x = -1 at bottom)
        self._x = -np.arange(1, self._nlayers + 1) / self._nlayers

        # Extended levels including canopy top
        self._xl = np.concatenate([[0], self._x])

        # Calculate roughness parameters if not specified
        if self.zo is None:
            # Simple approximation: zo ≈ 0.1 * hc
            object.__setattr__(self, "zo", 0.1 * self.hc)
        if self.d is None:
            # Simple approximation: d ≈ 0.7 * hc
            object.__setattr__(self, "d", 0.7 * self.hc)

    @property
    def nlayers(self) -> int:
        """Number of canopy layers."""
        return self._nlayers

    @property
    def litab(self) -> NDArray[np.float64]:
        """Leaf inclination angle classes [degrees]."""
        return self._litab

    @property
    def lazitab(self) -> NDArray[np.float64]:
        """Leaf azimuth angle classes [degrees]."""
        return self._lazitab

    @property
    def x(self) -> NDArray[np.float64]:
        """Cumulative canopy levels (0 to -1).

        Note: This is computed dynamically to ensure consistency with nlayers.
        """
        # Recompute based on current nlayers to ensure consistency
        return -np.arange(1, self._nlayers + 1) / self._nlayers

    @property
    def xl(self) -> NDArray[np.float64]:
        """Cumulative canopy levels including top (0 at top).

        Note: This is computed dynamically to ensure consistency with nlayers.
        """
        # Recompute based on current nlayers to ensure consistency
        x = -np.arange(1, self._nlayers + 1) / self._nlayers
        return np.concatenate([[0], x])

    @property
    def lidf(self) -> Optional[NDArray[np.float64]]:
        """Leaf inclination distribution function."""
        return self._lidf

    @lidf.setter
    def lidf(self, value: NDArray[np.float64]) -> None:
        """Set the LIDF array."""
        self._lidf = value

    @property
    def hot(self) -> float:
        """Hotspot parameter (leafwidth / canopy height)."""
        return self.leafwidth / self.hc if self.hc > 0 else 0.0

    @property
    def dx(self) -> float:
        """Layer thickness in relative LAI units."""
        return 1.0 / self._nlayers

    def copy(self, **kwargs) -> "Canopy":
        """Create a copy with optionally modified parameters.

        Args:
            **kwargs: Parameters to override in the copy.

        Returns:
            New Canopy instance with modified parameters.
        """
        from dataclasses import asdict, fields

        # Get only the public fields, not the private computed ones
        params = {f.name: getattr(self, f.name)
                  for f in fields(self)
                  if not f.name.startswith("_")}
        params.update(kwargs)
        return Canopy(**params)
